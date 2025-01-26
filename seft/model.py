import math
from pathlib import Path
import json
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix='SEFT_MODEL_',
        env_file='.env',
        extra="ignore"
    )
    d_model: int = Field(768)
    nhead: int = Field(12)
    n_channels: int = Field(1)
    dim_feedforward: int = Field(3072)
    activation: str = Field("gelu")
    hidden_dropout_prob: float = Field(0.1)
    attention_probs_dropout_prob: float = Field(0.1)
    initializer_range: float = Field(0.02)
    layer_norm_eps: float = Field(1e-12)
    max_len: int = Field(1500)
    drop_rate: float = Field(0.)
    drop_path_rate: float = Field(0.)
    depth: int = Field(12)
    mlp_ratio: float = Field(4.)
    attn_drop_rate: float = Field(0.)
    init_values: float = Field(0.)
    init_scale: float = Field(0.)
    head_drop_rate: float = Field(0.)
    dim_output: Optional[int] = Field(None)
    tubelet_size: tuple[int, int, int] = Field((1, 1, 16))
    pos_encoding: str = Field("relative")
    multiple_of: int = Field(2)


class RoPENd(torch.nn.Module):
    """N-dimensional Rotary Positional Embedding."""

    def __init__(self, n_dims, d_model, base=10000):
        super(RoPENd, self).__init__()

        k_max = d_model // (2 * n_dims)
        self.head_dim = d_model
        self.x_shape = None

        assert d_model % k_max == 0, f'shape[-1] ({d_model}) is not divisible by 2 * len(shape[:-1]) ({2 * n_dims})'

        self.buff = self.register_buffer("theta_ks", 1 / (
                    base ** (torch.arange(k_max) / k_max)))

    def build_angles(self, positions: list[torch.tensor]):
        # create a stack of angles multiplied by position
        angles = torch.stack([
            torch.cat(
                [t.unsqueeze(-1) * self.theta_ks for t in torch.meshgrid(
                    p, indexing='ij')], dim=-1) for p in zip(*positions)])
        # convert to complex number to allow easy rotation
        rotations = torch.polar(torch.ones_like(angles), angles)
        return rotations

    def set_x_shape(self, x_shape):
        self.x_shape = x_shape

    def forward(self, x, positions: torch.tensor):
        if self.x_shape is None:
            raise ValueError(
                "x_shape must be set before the first forward pass")
        B, N, H, E = x.shape
        cls = x[:, 0]
        # Reshape the input and ignore the CLS token
        x = x[:, 1:].view(B, *self.x_shape, H, E)

        # convert input into complex numbers to perform rotation
        x = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
        rotation = self.build_angles(positions)
        pe_x = rotation[..., None, :] * x
        x = torch.view_as_real(pe_x).view(B, N-1, H, E)
        return torch.cat((cls.unsqueeze(1), x), dim=1)


class SEFT(nn.Module):
    def __init__(self, config: ModelConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if config.d_model % 2 != 0:
            raise ValueError(f"Cannot use sin/cos positional encoding with "
                             f"an odd dim ({config.d_model})")
        self.config: ModelConfig = config
        self.pos_drop = nn.Dropout(p=config.drop_rate)
        self.n_layers = config.depth

        self.layers = torch.nn.ModuleList()
        for layer_id in range(self.n_layers):
            self.layers.append(TransformerBlock(layer_id, config))

        self.norm = RMSNorm(config.d_model, eps=config.layer_norm_eps)

        self.head_dropout = nn.Dropout(config.head_drop_rate)
        self.cls = nn.Parameter(torch.zeros(config.d_model))
        self.head = lambda x: x
        if config.dim_output is not None:
            self.head = ClassificationHead(config)

        proj_input_dim = config.tubelet_size[0] * config.tubelet_size[1] * config.tubelet_size[2] * config.n_channels
        self.projection = nn.Linear(proj_input_dim, config.d_model)

        match config.pos_encoding:
            case "absolute":
                self.pos_embedding = PositionalEncoding(
                    config.d_model,
                    config.drop_rate,
                    config.max_len
                )
            case "relative":
                self.pos_embedding = RoPENd(
                    3,
                    config.d_model // config.nhead
                )

        self.loss_fn = nn.CrossEntropyLoss()
        self.register_buffer('zeros', torch.zeros(1, dtype=torch.bool))
        self._init_weights(self)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, x):
        """
        Expected input format: BTHW
        Converts the input video-like format to a sequence of patches
        """
        B, T, C, H, W = x.shape

        T_p = T // self.config.tubelet_size[0]
        H_p = H // self.config.tubelet_size[1]
        W_p = W // self.config.tubelet_size[2]
        N = T_p * H_p * W_p
        x = x.reshape(B, T_p, H_p, W_p, *self.config.tubelet_size, C)
        return x.reshape(B, N, -1), (T_p, H_p, W_p)

    def forward(self, values, T=None, H=None, W=None, pad_mask=None, label=None):
        positions = [i for i in [T, H, W] if i is not None]
        B = values.shape[0]

        x, shape = self.patchify(values)
        self.pos_embedding.set_x_shape(shape)

        x = self.projection(x)

        if self.config.pos_encoding == "absolute":
            x = self.pos_embedding(x, positions)
            x = self.pos_drop(x)

        x = torch.cat((self.cls.expand(B, 1, -1), x), dim=1)

        if pad_mask is not None:
            pad_mask = torch.cat([self.zeros.expand(B, 1), pad_mask], dim=1)
            pad_mask = F.pad(pad_mask, (0, 1)).unsqueeze(1).unsqueeze(2)

            neg_inf = torch.full(
                (x.shape[1], x.shape[1]), float("-inf"), device=x.device
            )
            neg_inf[~pad_mask] = 0

        else:
            pad_mask = torch.zeros((x.shape[1], x.shape[1]), device=x.device)

        for layer in self.layers:
            x = layer(x, positions, self.pos_embedding, pad_mask)

        x = self.head_dropout(x)
        logits = self.head(x)

        loss = None
        if label is not None:
            loss = self.loss_fn(logits, label)

        return logits, loss

    def set_loss_fn(self, loss_fn):
        self.loss_fn = loss_fn

    def set_head(self, head):
        self.head = head

    @staticmethod
    def load_from_checkpoint(checkpoint_dir):
        checkpoint_dir = Path(checkpoint_dir)
        with open(checkpoint_dir/"model_config.json", "r") as f:
            config_json = json.load(f)

        config = ModelConfig(**config_json)
        model = SEFT(config)
        model.load_weights(checkpoint_dir)
        return model

    def load_weights(self, checkpoint_dir):
        state_dict = load_file(Path(checkpoint_dir)/"model.safetensors")
        self.load_state_dict(state_dict)


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class ClassificationHead(nn.Module):
    """Simple default head useful for classification.
    """
    def __init__(self, config: ModelConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head = nn.Sequential(
            RMSNorm(config.d_model, config.layer_norm_eps),
            nn.Linear(config.d_model, config.dim_output)
        )

    def forward(self, x):
        x = x[:, 0, :]
        return self.head(x)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    """Multi-head attention module."""

    def __init__(self, args: ModelConfig):
        """
        Initialize the Attention module.

        Args:
            args (ModelArgs): Model configuration parameters.
        """
        super().__init__()
        self.n_kv_heads = args.nhead
        self.head_dim = args.d_model // args.nhead

        self.wq = nn.Linear(
            args.d_model,
            args.nhead * self.head_dim,
            bias=False
        )
        self.wk = nn.Linear(
            args.d_model,
            self.n_kv_heads * self.head_dim,
            bias=False
        )
        self.wv = nn.Linear(
            args.d_model,
            self.n_kv_heads * self.head_dim,
            bias=False
        )
        self.wo = nn.Linear(
            args.nhead * self.head_dim,
            args.d_model,
            bias=False
        )

    def forward(
            self,
            x: torch.Tensor,
            positions: list[torch.tensor],
            pos_emb: RoPENd,
            mask: Optional[torch.Tensor],
    ):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            positions (list[torch.tensor]): List of positional indices.
            pos_emb (RoPENd): Positional embedding transformation.
            mask (torch.Tensor, optional): Attention mask tensor.

        Returns:
            torch.Tensor: Output tensor after attention.

        """
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        xq, xk = pos_emb(xq, positions), pos_emb(xk, positions)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = xk.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = xv.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(
            self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores,
                              values)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
            self,
            dim: int,
            hidden_dim: int,
            multiple_of: int
    ):
        """
        Initialize the FeedForward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
        """
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * (
                    (hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelConfig):
        """
        Initialize a TransformerBlock.

        Args:
            layer_id (int): Identifier for the layer.
            args (ModelArgs): Model configuration parameters.
        """
        super().__init__()
        self.n_heads = args.nhead
        self.dim = args.d_model
        self.head_dim = args.d_model // args.nhead
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.d_model,
            hidden_dim=4 * args.d_model,
            multiple_of=args.multiple_of
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.d_model, eps=args.layer_norm_eps)
        self.ffn_norm = RMSNorm(args.d_model, eps=args.layer_norm_eps)

    def forward(
            self,
            x: torch.Tensor,
            positions: int,
            pos_embed: RoPENd,
            mask: Optional[torch.Tensor],
    ):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            positions (list[torch.Tensor]): Positions for all tokens.
            pos_embed (RoPENd): Positional embedding transformation.
            mask (torch.Tensor, optional): Masking tensor for attention. Defaults to None.

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.

        """
        h = x + self.attention(
            self.attention_norm(x), positions, pos_embed, mask
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class PositionalEncoding(nn.Module):
    """
    Based on the original encodings used in the paper.

    References:
    PyTorch: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor, idxs: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
            idxs: Tensor, shape ``[batch_size, seq_len]``
        """
        x = x + self.pe[idxs]
        return self.dropout(x)

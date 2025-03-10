import math
from pathlib import Path
import json
from typing import Optional, Literal, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from pydantic import Field, BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

from robite.positional_embeddings import (
    RoPENd,
    AbsoluteSinCosine,
    DummyPosEmbedding,
)
from robite.utils import RMSNorm, patchify, POSITION_DTYPE


class EncoderConfig(BaseModel):
    """
    RoBiTE Transformer Encodre configuration values.
    """
    d_model: int = Field(768)
    nhead: int = Field(12)
    dim_feedforward: Optional[int] = Field(None)
    activation: str = Field("gelu")
    hidden_dropout_prob: float = Field(0.1)
    attention_probs_dropout_prob: float = Field(0.1)
    initializer_range: float = Field(0.02)
    layer_norm_eps: float = Field(1e-12)
    drop_rate: float = Field(0.)
    drop_path_rate: float = Field(0.)
    depth: int = Field(12)
    mlp_ratio: float = Field(4.)
    attn_drop_rate: float = Field(0.)
    init_values: float = Field(0.)
    init_scale: float = Field(0.)
    multiple_of: int = Field(2)


class RoBiTEBaseConfig(BaseSettings):
    """
    RoBiTE base configuration.
    """
    encoder_config: EncoderConfig
    pos_encoding: Literal["ropend", "absolute"] = Field("ropend")
    # Dropout to be applied to the positional encoding
    pos_drop: float = Field(0.)
    # Maximum length of an input, used when precomputing static positional
    # encodings.
    max_len: int = Field(1500)
    tubelet_size: tuple[int, int, int] = Field((1, 1, 16))
    n_channels: int = Field(1)
    head_drop_rate: float = Field(0.)


class RoBiTEConfig(RoBiTEBaseConfig):
    """
    Configuration for RoBiTE classifier.
    """
    model_config = SettingsConfigDict(
        env_prefix='ROBITE_BASIC_',
        env_file='.env',
        extra="ignore",
        env_nested_delimiter='__'
    )
    dim_output: Optional[int]


class InterpolationConfig(RoBiTEBaseConfig):
    """
    Configuration for RoBiTE interpolation.
    """
    model_config = SettingsConfigDict(
        env_prefix='ROBITE_INTERP_',
        env_file='.env',
        extra="ignore",
        env_nested_delimiter='__'
    )
    decoder_config: EncoderConfig
    mask_ratio: float = Field(.5)


def _get_inpt_pos_embedding(pos_encoding: str, d_model: int,
                            pos_drop: float, max_len: int) -> nn.Module:
    """
    Parse the config and return the relevant positional encoding function to
    be applied at the input.
    """
    match pos_encoding:
        case "absolute":
            return AbsoluteSinCosine(
                d_model=d_model,
                dropout=pos_drop,
                max_len=max_len
            )
        case _:
            return DummyPosEmbedding()


def _get_attn_pos_embedding(pos_encoding: str, d_model: int,
                            nhead: int, pos_drop: float) -> nn.Module:
    """
    Parse the config and return the relevant positional encoding function to
    be applied at each attention block.
    """
    match pos_encoding:
        case "ropend":
            return RoPENd(
                n_dims=3,
                d_model=d_model // nhead,
                dropout=pos_drop
            )
        case _:
            return DummyPosEmbedding()


def _init_weights(m):
    """
    Initialize weights, biases and normalization weights in a RoBiTE model.
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, RMSNorm):
        nn.init.constant_(m.weight, 1.0)


def _get_attn_mask(x_shape: tuple[int, ...], device,
                  pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Generate the attention mask based on an input pad mask. If there is
    no pad mask, the attention mask will be all zeros.
    Also adds an entry at the beginning for the CLS token.
    """
    if pad_mask is not None:
        attn_mask = torch.full(
            (x_shape[0], x_shape[1], x_shape[1]), float("-inf"),
            device=device
        )
        # The attention mask only needs to be applied to the columns (keys)
        attn_mask[~pad_mask] = 0
        attn_mask = attn_mask.permute([0, 2, 1])[:, None, ...]
    else:
        attn_mask = torch.zeros((1, x_shape[1], x_shape[1]), device=device)
    return attn_mask


def load_from_checkpoint(checkpoint_dir, model_cls, model_config):
    checkpoint_dir = Path(checkpoint_dir)
    with open(checkpoint_dir / "model_config.json", "r") as f:
        config_json = json.load(f)

    config = model_config(**config_json)
    model = model_cls(config)
    model.load_weights(checkpoint_dir)
    return model


class RoBiTEBase(nn.Module):
    """
    Base RoBiTE model class. Contains logic for positional encoding switching
    (relative vs absolute), weight initialization, patchification, and more.
    """
    def __init__(self, config: RoBiTEBaseConfig, d_model: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]
        self.loss_fn = None
        # All models use the exact same encoder backend
        self.encoder: nn.Module = Encoder(
            config=config.encoder_config
        )
        # Projection from tubelets to embedding dimension
        proj_input_dim = (
                config.tubelet_size[0] * config.tubelet_size[1] *
                config.tubelet_size[2] * config.n_channels
        )
        self.projection = nn.Linear(proj_input_dim, d_model)

        # Classification token
        self.register_parameter("cls", nn.Parameter(torch.zeros(d_model)))
        nn.init.trunc_normal_(self.cls, std=.02)

        # A useful zero buffer
        self.register_buffer("zeros", torch.zeros(1))
        self._inpt_pos_embedding, self._attn_pos_embedding = None, None


    @staticmethod
    def get_pos_embs(config: RoBiTEBaseConfig, nhead: int, d_model: int) -> tuple[nn.Module, nn.Module]:
        """Load positional embeddings based on the provided config.
        """
        # Positional embeddings applied before encoder forward pass, usually
        # absolute
        inpt_pos_embedding = _get_inpt_pos_embedding(
            pos_encoding=config.pos_encoding,
            d_model=d_model,
            pos_drop=config.pos_drop,
            max_len=config.max_len
        )
        # Positional embeddings applied within each attention block, usually
        # relative (RoPE)
        attn_pos_embedding = _get_attn_pos_embedding(
            pos_encoding=config.pos_encoding,
            d_model=d_model,
            nhead=nhead,
            pos_drop=config.pos_drop
        )
        return inpt_pos_embedding, attn_pos_embedding

    def set_loss_fn(self, loss_fn):
        """Manually set the model loss function
        """
        self.loss_fn = loss_fn

    def load_weights(self, checkpoint_dir):
        state_dict = load_file(Path(checkpoint_dir)/"model.safetensors")
        self.load_state_dict(state_dict)

    def get_loss(self, logits: torch.Tensor,  label: torch.Tensor) -> torch.Tensor:
        """Calculate loss if the label is not None
        """
        loss = None
        if label is not None:
            loss = self.loss_fn(logits, label)
        return loss

    def add_cls(self, x, positions, pad_mask):
        # Add classification token to the beginning of all relevant tensors
        x = torch.cat((self.cls.expand(x.shape[0], 1, -1), x), dim=1)
        positions = torch.cat([self.zeros.expand(x.shape[0], positions.shape[1], -1), positions], dim=2)
        pad_mask = torch.cat([(self.zeros > .5).expand(x.shape[0], -1), pad_mask], dim=1)
        return x, positions, pad_mask


class RoBiTEForInterpolation(RoBiTEBase):
    def __init__(self, config: InterpolationConfig, *args, **kwargs):
        super().__init__(config, config.encoder_config.d_model, *args, **kwargs)
        self.config: InterpolationConfig = config
        self.decoder = Encoder(config.decoder_config)
        # Projection from encoder embedding dimension to decoder
        # embedding dimension
        self.encoder_decoder_proj = nn.Linear(config.encoder_config.d_model,
                                              config.decoder_config.d_model)

        self.encoder_inpt_pos_embedding, self.encoder_attn_pos_embedding = self.get_pos_embs(
            config, nhead=config.encoder_config.nhead, d_model=config.encoder_config.d_model
        )
        self.decoder_inpt_pos_embedding, self.decoder_attn_pos_embedding = self.get_pos_embs(
            config, nhead=config.decoder_config.nhead, d_model=config.decoder_config.d_model
        )

        self.loss_fn = nn.MSELoss()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.decoder_config.d_model))
        self.head = InterpolationHead(
            d_model=config.decoder_config.d_model,
            d_output=math.prod(config.tubelet_size),
            layer_norm_eps=config.decoder_config.layer_norm_eps,
            head_drop_rate=config.head_drop_rate
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, values: torch.Tensor, mask: torch.Tensor,
                positions: POSITION_DTYPE, pad_mask=None,
                label=None) -> tuple[torch.Tensor, torch.Tensor]:
        b = values.shape[0]
        # Convert input to a sequence of tubelets
        x = patchify(self.config.tubelet_size, values)

        # Extract all the values that are being masked out
        # There is a little bit of index juggling because of the extra CLS
        # token which isn't in x yet
        m_x = x[mask].reshape(b, -1, x.shape[-1])
        m_positions = positions[mask[:, None, ].expand(-1, 3, -1)].reshape(b, 3, -1)
        m_pad_mask = pad_mask[mask].reshape(b, -1)

        # Now get all the values that are not masked out
        x = x[~mask].reshape(b, -1, x.shape[-1])
        positions = positions[~mask[:, None, ...].expand(-1, 3, -1)].reshape(b, 3, -1)
        pad_mask = pad_mask[~mask].reshape(b, -1)

        # Project all unmasked tubelets into embeddings
        x = self.projection(x)
        # Add classification token to the beginning of all relevant tensors
        x, positions, pad_mask = self.add_cls(x, positions, pad_mask)

        x = self.encoder_inpt_pos_embedding(x, positions)

        attn_mask = _get_attn_mask(x.shape, x.device, pad_mask)

        # Encoder forward pass
        x = self.encoder(
            x,
            positions=positions,
            pos_encoding=self.encoder_attn_pos_embedding,
            attn_mask=attn_mask
        )
        # Project tokens from the encoder dimension to decoder dimension
        x = self.encoder_decoder_proj(x)

        mask_tokens = self.mask_token.expand(b, m_x.shape[1], -1)

        # Apply input positional encodings to our MASK tokens.
        mask_tokens = self.encoder_inpt_pos_embedding(mask_tokens, m_positions)

        # Append MASK token and positional information
        x = torch.cat([x, mask_tokens], dim=1)
        positions = torch.cat([positions, m_positions], dim=2)
        pad_mask = torch.cat([pad_mask, m_pad_mask], dim=1)

        # Get our new attention and padding masks
        attn_mask = _get_attn_mask(x.shape, x.device, pad_mask)

        # Decoder forward pass
        x = self.decoder(
            x,
            positions=positions,
            pos_encoding=self.decoder_attn_pos_embedding,
            attn_mask=attn_mask
        )

        # Apply head to get logits and calculate loss
        logits = self.head(x[:, -m_x.shape[-2]:])
        loss = None
        if m_x.shape[1] != 0:
            loss = self.get_loss(logits, m_x)

        # We reset the positional embedding caches to avoid
        # inter-loop dependencies in the Trainer.
        self.encoder_inpt_pos_embedding.reset_cache()
        self.encoder_attn_pos_embedding.reset_cache()
        self.decoder_inpt_pos_embedding.reset_cache()
        self.decoder_attn_pos_embedding.reset_cache()

        return logits, loss


class RoBiTE(RoBiTEBase):
    """
    Basic RoBiTE model with an MLP head on top. Useful for regression and
    classification tasks.
    """
    def __init__(self, config: RoBiTEConfig, *args, **kwargs):
        super().__init__(config=config, d_model=config.encoder_config.d_model, *args, **kwargs)
        self.config: RoBiTEConfig = config
        self.head = ClassificationHead(
            d_model=config.encoder_config.d_model,
            d_output=config.dim_output,
            layer_norm_eps=config.encoder_config.layer_norm_eps,
            head_drop_rate=config.head_drop_rate
        )
        self.inpt_pos_embedding, self.attn_pos_embedding = self.get_pos_embs(
            config, nhead=config.encoder_config.nhead, d_model=config.encoder_config.d_model
        )
        self.apply(_init_weights)

    @staticmethod
    def from_pretrained(checkpoint: str, **kwargs):
        p_model = load_from_checkpoint(
            checkpoint, RoBiTEForInterpolation, InterpolationConfig
        )
        encoder_config = p_model.config.encoder_config
        finetune_config = RoBiTEConfig(
            encoder_config=encoder_config,
            pos_encoding=p_model.config.pos_encoding,
            tubelet_size=p_model.config.tubelet_size,
            n_channels=p_model.config.n_channels,
            max_len=p_model.config.max_len,
            **kwargs
        )
        model = RoBiTE(config=finetune_config)
        # Copy over all the encoder weights
        with torch.no_grad():
            p_model_state = p_model.state_dict()
            encoder_keys = {
                ".".join(key.split(".")[1:]): val for key, val in
                p_model_state.items() if key.split(".")[0] == "encoder"
            }
            model.encoder.load_state_dict(encoder_keys)
            model.cls.copy_(p_model.cls)
        return model

    def set_head(self, head):
        """Change the model head to whatever you want! Should accept the
        encoder output logits and return something that the loss can accept.
        """
        self.head = head

    def forward(self, values: torch.Tensor, positions: torch.Tensor,
                pad_mask=None, label=None) -> tuple[torch.Tensor, torch.Tensor]:
        # Convert input to a sequence of tubelets
        x = patchify(self.config.tubelet_size, values)

        x = self.projection(x)
        x, positions, pad_mask = self.add_cls(x, positions, pad_mask)
        attn_mask = _get_attn_mask(x.shape, x.device, pad_mask)
        x = self.inpt_pos_embedding(x, positions)
        # Encoder forward pass
        x = self.encoder(
            x,
            positions=positions,
            pos_encoding=self.attn_pos_embedding,
            attn_mask=attn_mask
        )

        # Apply head to get logits and calculate loss
        logits = self.head(x)
        loss = self.get_loss(logits, label)

        # We reset the positional embedding caches to avoid
        # inter-loop dependencies in the Trainer.
        self.inpt_pos_embedding.reset_cache()
        self.attn_pos_embedding.reset_cache()

        return logits, loss


class Encoder(nn.Module):
    def __init__(self, config: EncoderConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if config.d_model % 2 != 0:
            raise ValueError(f"Cannot use sin/cos positional encoding with "
                             f"an odd dim ({config.d_model})")
        self.config: EncoderConfig = config
        self.n_layers = config.depth

        self.layers = torch.nn.ModuleList()
        for layer_id in range(self.n_layers):
            self.layers.append(TransformerBlock(layer_id, config))

    def forward(self, x, positions, pos_encoding, attn_mask=None):
        for layer in self.layers:
            x = layer(x, positions, pos_encoding, attn_mask)
        return x


class ClassificationHead(nn.Module):
    """Simple default head utilizing the CLS token for classification.
    """
    def __init__(self, d_model: int, d_output: int, layer_norm_eps: float,
                 head_drop_rate: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head = nn.Sequential(
            nn.Dropout(head_drop_rate),
            RMSNorm(d_model, layer_norm_eps),
            nn.Linear(d_model, d_output)
        )

    def forward(self, x):
        # Take out the CLS token
        x = x[:, 0, :]
        return self.head(x)


class InterpolationHead(nn.Module):
    """Simple interpolation head making predictions on the original tubelet
    values from the learned MASK tokens.
    """
    def __init__(self, d_model: int, d_output: int, layer_norm_eps: float,
                 head_drop_rate: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dropout = nn.Dropout(head_drop_rate)
        self.head = nn.Sequential(
            RMSNorm(d_model, layer_norm_eps),
            nn.Linear(d_model, d_output)
        )

    def forward(self, x):
        # Apply dropout
        x = self.dropout(x)
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

    def __init__(self, config: EncoderConfig):
        """
        Initialize the Attention module.

        Args:
            config : EncoderConfig
        """
        super().__init__()
        self.n_kv_heads = config.nhead
        self.head_dim = config.d_model // config.nhead

        self.wq = nn.Linear(
            config.d_model,
            config.nhead * self.head_dim,
            bias=False
        )
        self.wk = nn.Linear(
            config.d_model,
            self.n_kv_heads * self.head_dim,
            bias=False
        )
        self.wv = nn.Linear(
            config.d_model,
            self.n_kv_heads * self.head_dim,
            bias=False
        )
        self.wo = nn.Linear(
            config.nhead * self.head_dim,
            config.d_model,
            bias=False
        )

    def forward(
            self,
            x: torch.Tensor,
            positions: Optional[list[torch.tensor]] = None,
            pos_emb: Optional[RoPENd] = None,
            mask: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass of the attention module.

        Args:
            x : torch.Tensor
            positions : List[torch.Tensor], optional
            pos_emb : Callable[[torch.Tensor], torch.Tensor], optional
            mask : torch.Tensor, optional

        Returns:
            x_out : torch.Tensor
                Output tensor after attention.
        """
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        if pos_emb is not None:
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
    def __init__(self, layer_id: int, config: EncoderConfig):
        """
        Initialize a TransformerBlock.

        Args:
            layer_id : int
                Identifier for the layer.
            config : EncoderConfig
                Model configuration parameters.
        """
        super().__init__()
        self.n_heads = config.nhead
        self.dim = config.d_model
        self.head_dim = config.d_model // config.nhead
        self.attention = Attention(config)

        hidden_dim = config.dim_feedforward if config.dim_feedforward is not None else 4 * config.d_model
        self.feed_forward = FeedForward(
            dim=config.d_model,
            hidden_dim=hidden_dim,
            multiple_of=config.multiple_of
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(config.d_model, eps=config.layer_norm_eps)
        self.ffn_norm = RMSNorm(config.d_model, eps=config.layer_norm_eps)

    def forward(
            self,
            x: torch.Tensor,
            positions: Optional[int] = None,
            pos_embed: Optional[RoPENd] = None,
            mask: Optional[torch.Tensor] = None
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

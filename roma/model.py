import math
from pathlib import Path
from typing import Optional, Literal, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from pydantic import Field, BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

from roma.positional_embeddings import (
    NDPRope,
    AbsoluteSinCosine,
    DummyPosEmbedding,
)
from roma.utils import get_drop_path, patchify, load_from_checkpoint, get_encoder_size

"""
RoMA architecture implementation.
The Transformer implementation is a modified version of the one provided by 
Llama:
https://github.com/meta-llama/llama
https://arxiv.org/abs/2302.13971

Some pieces are based on the implementation of the original MAE:
https://github.com/facebookresearch/mae
https//arxiv.org/abs/2111.06377 
"""

class EncoderConfig(BaseModel):
    """
    RoMA Encoder configuration values.
    """
    d_model: int = Field(342)
    nhead: int = Field(8)
    layer_norm_eps: float = Field(1e-12)
    depth: int = Field(6)
    # To manually set the dimension of the MLP, change dim_feedforward.
    dim_feedforward: Optional[int] = Field(None)
    # If dim_feedforward is None it's chosen by multiplying d_model by the mlp_ratio
    mlp_ratio: float = Field(4.)
    # Stochastic depth value
    drop_path_rate: float = Field(0.)
    # Dropout to be applied throughout the Transformer
    hidden_drop_rate: float = Field(0.)
    attn_proj_drop_rate: float = Field(0.)
    attn_drop_rate: float = Field(0.)
    pos_drop_rate: float = Field(0.)


class RoMABaseConfig(BaseSettings):
    """
    RoMA base configuration, shared by RoMAForClassification and
    RoMAForInterpolation.
    """
    encoder_config: EncoderConfig = Field(EncoderConfig())
    use_cls: bool = Field(
        True,
        description="Whether to insert a learned CLS token at the start of "
                    "the sequence."
    )
    pos_encoding: Literal["ropend", "absolute"] = Field("ropend")
    # Maximum length of an input, used when precomputing static positional
    # encodings.
    max_len: int = Field(1500)
    tubelet_size: tuple[int, int, int] = Field((1, 1, 16))
    n_channels: int = Field(1)
    head_drop_rate: float = Field(0.)
    n_pos_dims: int = Field(3)
    p_rope_val: float = Field(0.75)


class RoMAForClassificationConfig(RoMABaseConfig):
    """
    Configuration parameters for RoMAForClassification.
    """
    model_config = SettingsConfigDict(
        env_prefix='ROMA_CLASSIFIER_',
        env_file='.env',
        extra="ignore",
        env_nested_delimiter='__'
    )
    dim_output: Optional[int]


class RoMAForPreTrainingConfig(RoMABaseConfig):
    """
    Configuration parameters for the RoMAForPretraining model.
    """
    model_config = SettingsConfigDict(
        env_prefix='ROMA_PRETRAIN_',
        env_file='.env',
        extra="ignore",
        env_nested_delimiter='__'
    )
    decoder_config: EncoderConfig = Field(EncoderConfig(**get_encoder_size("RoMA-tiny-shallow")))
    normalize_targets: bool = Field(
        False,
        description="Whether to normalize the target tubelet values."
                    "Normalization is done per-tubelet, therefore this "
                    "setting should not be True when the tubelet size is "
                    "very small like (1, 1, 1)."
    )


def _get_inpt_pos_embedding(pos_encoding: str, d_model: int, max_len: int) -> nn.Module:
    """
    Parse the config and return the relevant positional encoding function to
    be applied at the input.
    Currently only supports the standard sin/cos absolute positional
    encodings. But is useful if we want to add more.
    """
    match pos_encoding:
        case "absolute":
            return AbsoluteSinCosine(
                d_model=d_model,
                max_len=max_len
            )
        case _:
            return DummyPosEmbedding()


def _get_attn_pos_embedding(pos_encoding: str, d_model: int, nhead: int, n_dims,
                            p: float) -> nn.Module:
    """
    Parse the config and return the relevant positional encoding function to
    be applied at each attention block.
    Currently only supports Continuous RopeND.
    """
    match pos_encoding:
        case "ropend":
            return NDPRope(
                n_dims=n_dims,
                head_dim=d_model // nhead,
                p=p
            )
        case _:
            return DummyPosEmbedding()


def _init_weights(m):
    """
    Initialize all weights, biases and normalization weights in the model.
    Used through self.apply(_init_weights) in the model init.
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.RMSNorm):
        nn.init.constant_(m.weight, 1.0)


def _get_attn_mask(x_shape: tuple[int, ...], device,
                   pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Generate an attention mask based on an input pad mask. If there is
    no pad mask, the attention mask will be all zeros.
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


class RoMABase(nn.Module):
    """
    Base RoMA model class. Contains common layers shared between all RoMA
    models.
    """
    def __init__(self, config: RoMABaseConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]
        self.loss_fn = None
        self.head = None

        self.inpt_pos_dropout = nn.Dropout(config.encoder_config.pos_drop_rate)

        # All models use the exact same encoder
        self.encoder: nn.Module = Encoder(
            config=config.encoder_config
        )
        # Projection from tubelets to embedding dimension
        proj_input_dim = (
                config.tubelet_size[0] * config.tubelet_size[1] *
                config.tubelet_size[2] * config.n_channels
        )
        self.projection = nn.Linear(proj_input_dim, config.encoder_config.d_model)

        # Classification token
        if config.use_cls:
            self.register_parameter(
                "cls",
                nn.Parameter(torch.zeros(config.encoder_config.d_model))
            )
            nn.init.trunc_normal_(self.cls, std=.02)
        else:
            self.cls = None

        # A useful zero buffer
        self.register_buffer("zeros", torch.zeros(1))

    def apply_head_loss(self, x, label: torch.Tensor | None):
        """Apply head and calculate loss
        """
        logits = self.head(x)
        loss = self.get_loss(logits, label)
        return logits, loss

    def get_pos_embs(self, config: RoMABaseConfig, nhead: int, d_model: int) -> tuple[nn.Module, nn.Module]:
        """Load positional embeddings based on the provided config and add
        dropout to them.
        """
        inpt_pos_embedding = _get_inpt_pos_embedding(
                pos_encoding=config.pos_encoding,
                d_model=d_model,
                max_len=config.max_len
        )
        attn_pos_embedding = _get_attn_pos_embedding(
            pos_encoding=config.pos_encoding,
            d_model=d_model,
            nhead=nhead,
            n_dims=config.n_pos_dims,
            p=config.p_rope_val
        )
        return inpt_pos_embedding, attn_pos_embedding

    def set_loss_fn(self, loss_fn):
        """
        Manually set the model loss function. The loss function should accept
        prediction and target tensors.
        """
        self.loss_fn = loss_fn

    def set_head(self, head):
        """
        Manually set the model head. The head accepts the raw output embeddings
        from the final layer of the model and returns something that can be fed
        into the loss function.
        """
        self.head = head

    def load_weights(self, checkpoint_dir):
        """Load model weights from a directory.
        """
        state_dict = load_file(Path(checkpoint_dir)/"model.safetensors")
        self.load_state_dict(state_dict, strict=False)

    def get_loss(self, logits: torch.Tensor,  label: torch.Tensor) -> torch.Tensor:
        """Call the loss function if the label is not None
        """
        loss = None
        if label is not None:
            loss = self.loss_fn(logits, label)
        return loss

    def add_cls(self, x, positions, pad_mask=None):
        if self.cls is not None:
            # Add classification token to the beginning of all relevant tensors
            x = torch.cat((self.cls.expand(x.shape[0], 1, -1), x), dim=1)
            positions = torch.cat([self.zeros.expand(x.shape[0], positions.shape[1], -1), positions], dim=2)
            if pad_mask is not None:
                pad_mask = torch.cat([(self.zeros > .5).expand(x.shape[0], -1), pad_mask], dim=1)
        return x, positions, pad_mask


class RoMAForPreTraining(RoMABase):
    def __init__(self, config: RoMAForPreTrainingConfig, *args, **kwargs):
        super().__init__(config,  *args, **kwargs)
        self.config: RoMAForPreTrainingConfig = config
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
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.decoder_config.d_model))
        self.set_head(
            InterpolationHead(
                d_model=config.decoder_config.d_model,
                d_output=math.prod(config.tubelet_size)*config.n_channels,
                layer_norm_eps=config.decoder_config.layer_norm_eps,
                head_drop_rate=config.head_drop_rate
        ))
        self.set_loss_fn(nn.MSELoss())

    def reset_pos_cache(self):
        self.encoder_inpt_pos_embedding.reset_cache()
        self.encoder_attn_pos_embedding.reset_cache()
        self.decoder_inpt_pos_embedding.reset_cache()
        self.decoder_attn_pos_embedding.reset_cache()

    def normalize_targets(self, x):
        """
        Normalize the input values. Normalization is applied individually
        for each tubelet, therefore this would be invalid for a tubelet
        size of (1, 1, 1), as all values would just be zero.
        """
        if self.config.normalize_targets:
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True)
            x = (x - mean) / (var + 1.e-6) ** .5
        return x

    def forward(self, values: torch.Tensor, mask: torch.Tensor,
                positions: torch.Tensor, pad_mask=None,
                label=None, *_, **__) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        b = values.shape[0]
        npd = self.config.n_pos_dims
        # Convert input to a sequence of tubelets
        x = patchify(self.config.tubelet_size, values)

        # Extract all the values that are being masked out
        m_x = x[mask].reshape(b, -1, x.shape[-1])
        m_positions = positions[mask[:, None, ].expand(-1, npd, -1)].reshape(b, npd, -1)
        if pad_mask is not None:
            m_pad_mask = pad_mask[mask].reshape(b, -1)

        # Now get all the values that are not masked out
        x = x[~mask].reshape(b, -1, x.shape[-1])
        positions = positions[~mask[:, None, ...].expand(-1, npd, -1)].reshape(b, npd, -1)
        if pad_mask is not None:
            pad_mask = pad_mask[~mask].reshape(b, -1)

        # Project into embeddings
        x = self.projection(x)
        # Add classification token to the beginning of all relevant tensors
        x, positions, pad_mask = self.add_cls(x, positions, pad_mask)

        x = self.inpt_pos_dropout(self.encoder_inpt_pos_embedding(x, positions))

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
        if pad_mask is not None:
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
        m_x = self.normalize_targets(m_x)
        x = x[:, -m_x.shape[-2]:]

        logits, loss = None, None
        if m_x.shape[1] != 0:
            logits, loss = self.apply_head_loss(x, m_x)

        # We reset the positional embedding caches to avoid
        # inter-loop dependencies in the Trainer, which break torch compile.
        self.reset_pos_cache()

        return logits, loss

    def predict(self, values: torch.Tensor, positions: torch.Tensor,
                pred_positions: torch.Tensor, pad_mask=None, pred_pad_mask=None) -> torch.Tensor:
        """
        Take a set of values and their positions and do predictions on a
        different set of positions.

        Parameters
        ----------
        values : torch.Tensor
            Known sequence values
        positions : torch.Tensor
            Positions corresponding to the values
        pred_positions : torch.Tensor
            A set of positions to predict
        pad_mask : torch.Tensor
            A pad mask with true in positions that are padded
        pred_pad_mask : torch.Tensor
            Same as pad mask but for the pred_positions

        Returns
        -------
        logits : torch.Tensor
            Model output logits for the predicted tokens
        """
        b = values.shape[0]
        # Convert input to a sequence of tubelets
        x = patchify(self.config.tubelet_size, values)
        # Project into embeddings
        x = self.projection(x)
        # Add classification token to the beginning of all relevant tensors
        x, positions, pad_mask = self.add_cls(x, positions, pad_mask)
        # Apply positional embeddings and dropout
        x = self.inpt_pos_dropout(self.encoder_inpt_pos_embedding(x, positions))
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
        # Create a sequence of mask tokens representing the values we want to
        # predict
        mask_tokens = self.mask_token.expand(b, pred_positions.shape[2], -1)
        mask_tokens = self.encoder_inpt_pos_embedding(mask_tokens, pred_positions)

        # Append MASK tokens to the original sequence
        x = torch.cat([x, mask_tokens], dim=1)
        positions = torch.cat([positions, pred_positions], dim=2)
        if pad_mask is not None and pred_pad_mask is not None:
            pad_mask = torch.cat([pad_mask, pred_pad_mask], dim=1)

        # Get our new attention and padding masks
        attn_mask = _get_attn_mask(x.shape, x.device, pad_mask)

        # Decoder forward pass
        x = self.decoder(
            x,
            positions=positions,
            pos_encoding=self.decoder_attn_pos_embedding,
            attn_mask=attn_mask
        )
        # Extract our predicted values
        x = x[:, -pred_positions.shape[-1]:]

        logits = None
        if pred_positions.shape[1] != 0:
            logits = self.head(x)

        self.reset_pos_cache()
        return logits



class RoMAForClassification(RoMABase):
    """
    Basic RoMA Encoder model with an MLP head on top. Useful for regression and
    classification tasks. Usually you want to initialize this using pre-trained
    weights from RoMAForPreTraining.
    """
    def __init__(self, config: RoMAForClassificationConfig, *args, **kwargs):
        super().__init__(config=config, *args, **kwargs)
        self.config: RoMAForClassificationConfig = config
        if config.use_cls:
            self.set_head(
                CLSClassifierHead(
                    d_model=config.encoder_config.d_model,
                    d_output=config.dim_output,
                    layer_norm_eps=config.encoder_config.layer_norm_eps,
                    head_drop_rate=config.head_drop_rate
            ))
        else:
            self.set_head(
                MeanClassifierHead(
                    d_model=config.encoder_config.d_model,
                    d_output=config.dim_output,
                    layer_norm_eps=config.encoder_config.layer_norm_eps,
                    head_drop_rate=config.head_drop_rate
                )
            )
        self.set_loss_fn(nn.CrossEntropyLoss())
        self.inpt_pos_embedding, self.attn_pos_embedding = self.get_pos_embs(
            config, nhead=config.encoder_config.nhead, d_model=config.encoder_config.d_model
        )
        self.apply(_init_weights)

    @staticmethod
    def from_pretrained(checkpoint: str, **kwargs):
        """
        Initialize the model from a pre-trained checkpoint created by
        RoMAForPreTraining.
        Parameters such as dropout are not brought over to allow for
        overriding during fine-tuning.
        """
        p_model = load_from_checkpoint(
            checkpoint, RoMAForPreTraining, RoMAForPreTrainingConfig
        )
        encoder_config = p_model.config.encoder_config

        finetune_config = RoMAForClassificationConfig(
            pos_encoding=p_model.config.pos_encoding,
            tubelet_size=p_model.config.tubelet_size,
            n_channels=p_model.config.n_channels,
            p_rope_val=p_model.config.p_rope_val,
            n_pos_dims=p_model.config.n_pos_dims,
            use_cls=p_model.config.use_cls,
            max_len=p_model.config.max_len,
            **kwargs
        )
        encoder_attrs = ["d_model", "nhead", "depth", "dim_feedforward",
                         "layer_norm_eps", "mlp_ratio"]
        for attr in encoder_attrs:
            setattr(finetune_config.encoder_config, attr, getattr(encoder_config, attr))

        model = RoMAForClassification(config=finetune_config)
        # Copy over all the encoder weights
        with torch.no_grad():
            p_model_state = p_model.state_dict()
            encoder_keys = {
                ".".join(key.split(".")[1:]): val for key, val in
                p_model_state.items() if key.split(".")[0] == "encoder"
            }
            model.encoder.load_state_dict(encoder_keys)
            if model.cls is not None:
                model.cls.copy_(p_model.cls)
        return model

    def reset_pos_cache(self):
        self.inpt_pos_embedding.reset_cache()
        self.attn_pos_embedding.reset_cache()

    def forward(self, values: torch.Tensor, positions: torch.Tensor,
                pad_mask=None, label=None, *_, **__) -> tuple[torch.Tensor, torch.Tensor]:
        x = patchify(self.config.tubelet_size, values)
        x = self.projection(x)
        x, positions, pad_mask = self.add_cls(x, positions, pad_mask)
        attn_mask = _get_attn_mask(x.shape, x.device, pad_mask)
        x = self.inpt_pos_embedding(x, positions)
        x = self.encoder(
            x,
            positions=positions,
            pos_encoding=self.attn_pos_embedding,
            attn_mask=attn_mask
        )
        logits, loss = self.apply_head_loss(x, label)

        self.reset_pos_cache()
        return logits, loss


class CLSClassifierHead(nn.Module):
    """Simple default head utilizing the CLS token for classification.
    """
    def __init__(self, d_model: int, d_output: int, layer_norm_eps: float,
                 head_drop_rate: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head = nn.Sequential(
            nn.Dropout(head_drop_rate),
            nn.RMSNorm(d_model, layer_norm_eps),
            nn.Linear(d_model, d_output)
        )

    def forward(self, x):
        # Take out the CLS token which is at position zero
        x = x[:, 0, :]
        return self.head(x)


class MeanClassifierHead(nn.Module):
    """Simple default head utilizing the mean of all tokens for classification.
    """
    def __init__(self, d_model: int, d_output: int, layer_norm_eps: float,
                 head_drop_rate: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head = nn.Sequential(
            nn.Dropout(head_drop_rate),
            nn.RMSNorm(d_model, layer_norm_eps),
            nn.Linear(d_model, d_output)
        )

    def forward(self, x):
        # Get the mean
        x = x.mean(dim=1)
        return self.head(x)


class InterpolationHead(nn.Module):
    """Simple interpolation head making predictions on the original tubelet
    values from the learned MASK tokens.
    """
    def __init__(self, d_model: int, d_output: int, layer_norm_eps: float,
                 head_drop_rate: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head = nn.Sequential(
            nn.Dropout(head_drop_rate),
            nn.RMSNorm(d_model, layer_norm_eps),
            nn.Linear(d_model, d_output)
        )

    def forward(self, x):
        return self.head(x)


class Encoder(nn.Module):
    """Transformer encoder module.
    """
    def __init__(self, config: EncoderConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config: EncoderConfig = config
        self.n_layers = config.depth

        self.layers = torch.nn.ModuleList()
        for layer_id in range(self.n_layers):
            self.layers.append(TransformerBlock(layer_id, config))

    def forward(self, x, positions, pos_encoding, attn_mask=None):
        for layer in self.layers:
            x = layer(x, positions, pos_encoding, attn_mask)
        return x


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
        self.attn_dropout_val = config.attn_drop_rate
        self.proj_dropout = nn.Dropout(config.attn_proj_drop_rate)
        self.pos_dropout = nn.Dropout(config.pos_drop_rate)
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
            pos_emb: Optional[NDPRope] = None,
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
            xq, xk = self.pos_dropout(pos_emb(xq, positions)), self.pos_dropout(pos_emb(xk, positions))

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = xk.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = xv.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        output = F.scaled_dot_product_attention(
            xq, keys, values,
            attn_mask=mask,
            dropout_p=self.attn_dropout_val if self.training else 0.
        )
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.proj_dropout(self.wo(output))


class FeedForward(nn.Module):
    def __init__(
            self,
            dim: int,
            hidden_dim: int,
            dropout: float
    ):
        """
        Initialize the FeedForward module.

        Args:
            dim : int
                Input dimension
            hidden_dim : int
                Hidden dimension of the feedforward layer.
            dropout : float
        """
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x))))


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
        self.drop_path = get_drop_path(config.drop_path_rate, layer_id, config.depth)

        hidden_dim = config.dim_feedforward if config.dim_feedforward is not None else round(config.mlp_ratio * config.d_model)
        self.feed_forward = FeedForward(
            dim=config.d_model,
            hidden_dim=hidden_dim,
            dropout=config.hidden_drop_rate
        )
        self.layer_id = layer_id
        self.attention_norm = nn.RMSNorm(config.d_model, eps=config.layer_norm_eps)
        self.ffn_norm = nn.RMSNorm(config.d_model, eps=config.layer_norm_eps)

    def forward(
            self,
            x: torch.Tensor,
            positions: Optional[int] = None,
            pos_embed: Optional[NDPRope] = None,
            mask: Optional[torch.Tensor] = None
    ):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x : torch.Tensor
                Input tensor.
            positions : torch.Tensor
                Positions for all tokens.
            pos_embed : RoPENd, optional
                Positional embedding transformation.
            mask : torch.Tensor, optional
                Masking tensor for attention. Defaults to None.

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.

        """
        h = x + self.drop_path(self.attention(
            self.attention_norm(x), positions, pos_embed, mask
        ))
        out = h + self.drop_path(self.feed_forward(self.ffn_norm(h)))
        return out

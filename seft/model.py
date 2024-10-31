import torch
import torch.nn as nn
import numpy as np
import math

from torch.nn.functional import layer_norm


class SEFTConfig:
    def __init__(self,
                 tubelet_size: tuple[int, int, int] = (1, 1, 16),
                 d_model: int = 512,
                 num_layers: int = 12,
                 nhead: int = 12,
                 dim_feedforward: int = 3072,
                 activation: str = "gelu",
                 hidden_dropout_prob: float = 0.1,
                 attention_probs_dropout_prob: float = 0.1,
                 initializer_range: float = 0.02,
                 layer_norm_eps: float = 1e-12,
                 *args, **kwargs):
        self.tubelet_size = tubelet_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.activation = activation
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps


class SEFT(nn.Module):
    def __init__(self, config: SEFTConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if config.d_model % 2 != 0:
            raise ValueError(f"Cannot use sin/cos positional encoding with "
                             f"an odd dim ({config.d_model})")

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.hidden_dropout_prob,
            activation=config.activation,
            layer_norm_eps=config.layer_norm_eps,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
        )

    def forward(self, x, mask):
        x = self.patchify(x)
        x = x[:, mask]

        x = torch.nested.nested_tensor(x[:, mask], layout=torch.jagged)

        return x

    def patchify(self, x):
        """
        Expected input format: BTHW
        Converts the input video-like format to patches
        """
        B, T, H, W = x.shape[0], x.shape[1], x.shape[2], x.shape[3]

        T_p = T//self.tubelet_size[0]
        H_p = H//self.tubelet_size[1]
        W_p = W//self.tubelet_size[2]
        N = T_p * H_p * W_p

        return x.reshape(-1, N, T_p, H_p, W_p)

# Based on the following implementation:
# https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
def partial_positional_encoding(d_model, positions):
    """
    Generate a position embedding for a set of positions. Function avoids
    generating the whole positional encoding array and instead only generates
    what is needed.

    d_model : int
        Embedding dimension of the model
    positions : torch.Tensor
        Positions for which to generate the embeddings
    """
    length = positions.shape[1]
    pe = torch.zeros(length, d_model)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                          -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(positions.float() * div_term)
    pe[:, 1::2] = torch.cos(positions.float() * div_term)

    return pe

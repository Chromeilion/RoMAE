from abc import ABC
import math
from typing import Optional

import torch
import torch.nn as nn


# The 3D position of all embeddings is represented as a tensor with 3 rows.
POSITIONS = torch.Tensor


class BasePosEmbedding(ABC):
    """Abstract base class from which positional embeddings should inherit.
    """
    def reset_cache(self):
        ...


class DummyPosEmbedding(nn.Module, BasePosEmbedding):
    """To be used as a "None" positional embedding.
    """
    @staticmethod
    def forward(x, *_, **__):
        return x


class RoPENd(nn.Module, BasePosEmbedding):
    """N-dimensional Rotary Positional Embedding.
    """
    def __init__(self, n_dims, d_model, base=10000, dropout=0.):
        super(RoPENd, self).__init__()
        k_max = d_model // (2 * n_dims)
        self.head_dim = d_model
        self.subdim = d_model // n_dims
        self.dropout = nn.Dropout(dropout)

        assert d_model % k_max == 0, f'shape[-1] ({d_model}) is not divisible by 2 * len(shape[:-1]) ({2 * n_dims})'

        self.register_buffer(
            "theta_ks",
            (1 / (base ** (torch.arange(k_max) / k_max))).float()
        )

        self.prev_positions = None
        self.cache = None

    def reset_cache(self):
        self.cache = None
        self.prev_positions = None

    def build_angles(self, B, positions: POSITIONS):
        if self.prev_positions is not None:
            if positions.shape == self.prev_positions.shape:
                if torch.all(positions == self.prev_positions):
                    return self.cache

        freqs = torch.matmul(
            positions.unsqueeze(3),
            self.theta_ks[None, None, ...].expand(B, -1, -1).unsqueeze(2)).permute(0, 2, 1, 3).reshape(B, positions.shape[-1], -1)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64

        self.cache = freqs_cis
        self.prev_positions = positions
        return freqs_cis

    def forward(self, x, positions: torch.tensor):
        B, N, H, E = x.shape

        rotations = self.build_angles(B, positions)

        # convert input into complex numbers to perform rotation
        x = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))

        x = x * rotations[..., None, :]
        x = torch.view_as_real(x).reshape(B, N, H, E)

        return self.dropout(x)


class AbsoluteSinCosine(nn.Module, BasePosEmbedding):
    """
    Based on the original encodings used in the paper.

    References:
    PyTorch: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """
    def __init__(self, d_model: int, dropout: float = 0., max_len: int = 2000):
        super(AbsoluteSinCosine, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor, idxs: POSITIONS) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
            idxs: Tensor, shape ``[batch_size, seq_len]``
        """
        B = x.shape[0]
        # Dumb way of doing this but it works and we only need to do it
        # once.
        t = idxs[0]
        t = t[..., None] + idxs[1][:, None, :]
        t = t[..., None] + idxs[2][:, None, None, :]
        x = x + self.pe[t.view(B, -1)]
        return self.dropout(x)

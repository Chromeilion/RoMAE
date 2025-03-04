from abc import ABC
import math
from typing import Optional

import torch
import torch.nn as nn


# The 3D position of all embeddings is represented as 3 lists.
POSITIONS = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class BasePosEmbedding(ABC):
    """Abstract base class from which positional embeddings should inherit.
    """
    def __init__(self):
        self.x_shape: Optional[torch.Tensor] = None

    def set_x_shape(self, shape: torch.Tensor) -> None:
        if shape.shape != torch.Size([3]):
            raise ValueError(
                f"Expected shape to be [3], got {shape.shape}"
            )
        self.x_shape = shape


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

        self.buff = self.register_buffer("theta_ks", 1 / (
                    base ** (torch.arange(k_max) / k_max)))

        self.prev_positions = None
        self.cache = None

    def build_angles(self, B, positions: POSITIONS):
        if self.prev_positions is not None:
            if all(torch.equal(i, j) for i, j in zip(positions, self.prev_positions)):
                return self.cache

        rotations = []
        for i in positions:
            freqs = torch.bmm(i.unsqueeze(2), self.theta_ks[None, ...].repeat(B, 1).unsqueeze(1)).float()
            freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
            rotations.append(freqs_cis)

        rotations = torch.cat(rotations, dim=2)

        self.cache = rotations
        self.prev_positions = positions
        return rotations

    def forward(self, x, positions: torch.tensor):
        if self.x_shape is None:
            raise ValueError(
                "x_shape must be set before the first forward pass")
        B, N, H, E = x.shape

        # Take out the CLS token
        cls = x[:, 0]


        rotations = self.build_angles(B, positions)

        # convert input into complex numbers to perform rotation
        x = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))

        # Ignore the CLS token when applying positional information
        # Also add a dummy dim to broadcast with all heads
        x_ = x[:, 1:, ...] * rotations[..., None, :]
        x[:, 1:, ...] = x_
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

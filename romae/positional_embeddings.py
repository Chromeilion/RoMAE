from abc import ABC
import math

import torch
import torch.nn as nn


# The ND position of all embeddings is represented as a tensor with N rows.
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


class AbsoluteSinCosine(nn.Module, BasePosEmbedding):
    """
    Based on the original encodings used in the paper.

    References:
    PyTorch: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """
    def __init__(self, d_model: int, max_len: int = 2000):
        super(AbsoluteSinCosine, self).__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor, positions, mask: torch.tensor = None) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
            mask: Tensor, shape ``[batch_size, seq_len]``
        """
        if mask is None:
            return x + self.pe[None, :x.shape[1]].expand(x.shape[0], -1, -1)
        if mask[0].sum() != x.shape[1]:
            mask = torch.cat([torch.ones(x.shape[0], 1, device=mask.device, dtype=torch.bool), mask], dim=1)
            x + self.pe[None, :mask.shape[1], :].expand(x.shape[0], -1, -1)[
                mask].reshape(x.shape[0], x.shape[1], -1)
        return x + self.pe[None, :mask.shape[1], :].expand(x.shape[0], -1, -1)[mask].reshape(x.shape[0], x.shape[1], -1)


class NDPRope(nn.Module,  BasePosEmbedding):
    """
    N-dimensional continuous p-RoPE. The initial p-RoPE code was converted from
    the JAX implementation here:
    "Round and Round We Go! What Makes Rotary Positional Encodings Useful?"
    https://openreview.net/forum?id=GtvuNrk58a
    """
    def __init__(self, head_dim: int, base=10000, p=1, n_dims: int=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if head_dim % n_dims != 0:
            raise AttributeError(f"The head dimension ({head_dim}) is not "
                                 f"divisible by the number of positional axis ({n_dims})!")
        if 0 > p or p > 1:
            raise AttributeError(f"Provided p value ({p}) is not between 0 and 1!")

        self.axis_dim = head_dim // n_dims
        self.n_dims = n_dims

        rope_angles = int(p * self.axis_dim // 2)
        nope_angles = self.axis_dim // 2 - rope_angles

        fraction = 2. * torch.arange(0, rope_angles) / self.axis_dim
        self.register_buffer("timescale", nn.Parameter(nn.functional.pad(
            base ** fraction,
            (0, nope_angles),
            mode='constant',
            value=torch.inf
        )))

        self.cache = None

    def reset_cache(self):
        self.cache = None

    def get_sin_cos(self, positions):
        sinusoid_inp = (
                positions[..., torch.newaxis] / self.timescale[torch.newaxis,
                                                torch.newaxis, :]
        )
        sinusoid_inp = sinusoid_inp[..., torch.newaxis, :]
        sin = torch.sin(sinusoid_inp)
        cos = torch.cos(sinusoid_inp)

        return sin, cos

    def apply_ndprope(self, x, angles):
        sin, cos = angles
        first_half, second_half = torch.tensor_split(x, 2, dim=-1)
        first_part = first_half * cos - second_half * sin
        second_part = second_half * cos + first_half * sin
        out = torch.concatenate([first_part, second_part], dim=-1)
        return out.to(x.dtype)

    def forward(self, x, positions):
        """
        Parameters
        ----------
        x
        positions : Tensor, shape ``[batch_size, ndim, seq_len]``
            For 3D position, this would be ```[batch_size, 3, seq_len]```.
        """
        B, seq_len, nhead, head_dim = x.shape

        if self.cache is None:
            self.cache = []
            for i in range(self.n_dims):
                self.cache.append(self.get_sin_cos(positions[:, i].reshape(B, -1)))

        views = []
        for i in range(self.n_dims):
            views.append(self.apply_ndprope(
                x[..., self.axis_dim*i:self.axis_dim*(i+1)],
                self.cache[i]
            ))
        x = torch.cat(views, dim=-1)
        return x

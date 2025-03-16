from typing import Optional
import random
import json
from pathlib import Path

import torch
import torch.nn as nn
import math

POSITION_DTYPE = tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]


def patchify(tubelet_size: tuple[int, int, int], x):
    """
    Expected input format: BTCHW
    E.g. a regular video could be (1, 32, 3, 244, 244).
    Converts the input into a sequence of tubelets.
    """
    b, t, c, h, w = x.shape

    t_p = t // tubelet_size[0]
    h_p = h // tubelet_size[1]
    w_p = w // tubelet_size[2]
    n = t_p * h_p * w_p
    x = x.reshape(b, t_p, h_p, w_p, *tubelet_size, c)
    return x.reshape(b, n, -1)


class CosineLRScheduleWithWarmup(torch.optim.lr_scheduler.LRScheduler):
    """
    Cosine learning rate schedule with a linear warmup.
    """
    def __init__(self, optimizer, warmup_steps, total_steps):
        if warmup_steps > total_steps:
            raise ValueError("Warmup steps must be less than total steps")

        self.warmup_start_factor = 1e-5
        self.warmup_end_factor = 1
        self.total_warmup_iters = warmup_steps
        self.eta_min = 0.0
        self.T_max = total_steps
        super().__init__(optimizer)

    def get_lr_warmup(self):
        """Compute the learning rate."""
        if self.last_epoch == 0:
            return [
                group["lr"] * self.warmup_start_factor for group in self.optimizer.param_groups
            ]

        if self.last_epoch > self.total_warmup_iters:
            return [group["lr"] for group in self.optimizer.param_groups]

        return [
            group["lr"]
            * (
                1.0
                + (self.warmup_end_factor - self.warmup_start_factor)
                / (
                    self.total_warmup_iters * self.warmup_start_factor
                    + (self.last_epoch - 1) * (self.warmup_end_factor - self.warmup_start_factor)
                )
            )
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr_warmup(self):
        return [
            base_lr
            * (
                self.warmup_start_factor
                + (self.warmup_end_factor - self.warmup_start_factor)
                * min(self.total_warmup_iters, self.last_epoch)
                / self.total_warmup_iters
            )
            for base_lr in self.base_lrs
        ]

    def get_lr(self) -> list[float]:
        """Compute learning rate using chainable form of the scheduler."""
        torch.optim.lr_scheduler._warn_get_lr_called_within_step(self)
        if self._step_count < self.total_warmup_iters:
            return self.get_lr_warmup()
        return self.get_lr_cosine()

    def get_lr_cosine(self):
        """Retrieve the learning rate of each parameter group."""
        if self.last_epoch == 0:
            return [group["lr"] for group in self.optimizer.param_groups]
        elif self._step_count == 1 and self.last_epoch > 0:
            return [
                self.eta_min
                + (base_lr - self.eta_min)
                * (1 + math.cos((self.last_epoch) * math.pi / self.T_max))
                / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        elif (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [
                group["lr"]
                + (base_lr - self.eta_min) * (1 - math.cos(math.pi / self.T_max)) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        return [
            (1 + math.cos(math.pi * self.last_epoch / self.T_max))
            / (1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max))
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr_cosine(self):
        return [
            self.eta_min
            + (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * self.last_epoch / self.T_max))
            / 2
            for base_lr in self.base_lrs
        ]


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim : int
                The dimension of the input tensor.
            eps : float, optional
                A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps : float
                A small value added to the denominator for numerical stability.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x : torch.Tensor
                The input tensor.

        Returns:
            out : torch.Tensor
                The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x : torch.Tensor
                The input tensor.

        Returns:
            output : torch.Tensor
                The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def prepare_positions(b, positions: POSITION_DTYPE):
    """
    Take in a bunch of potentially None positions and replace None's
    with position zero (equating to no position in RoPE).
    This ensures that all position tensors have the same length
    (corresponding to the number of tokens).
    Additionally, move all positions forward by 1 and give the CLS token
    position zero.
    """
    if all([i is None for i in positions]):
        raise AttributeError("All position tensors cannot be None, set at "
                             "least one to a valid value!")
    n_positions = 0
    device = None
    for i in positions:
        if i is not None:
            n_positions = i.shape[1]
            device = i.device
            break
    pos = []
    for i, p in enumerate(positions):
        if p is None:
            pos.append(torch.zeros((b, n_positions), device=device))
        else:
            pos.append(p+1)
    pos = torch.stack(pos).permute(1, 0, 2)
    return pos


def load_from_checkpoint(checkpoint_dir, model_cls, model_config):
    """
    Load a model from a checkpoint.

    Parameters
    ----------
    checkpoint_dir : str
    model_cls
        The actual uninitialized class of the model being loaded
    model_config
        Uninitialized configuration class of the model being loaded

    Returns
    -------
    model
        The provided model_cls loaded with the weights and
        configuration present in the checkpoint directory.
    """
    checkpoint_dir = Path(checkpoint_dir)
    # Load model configuration
    with open(checkpoint_dir/"model_config.json", "r") as f:
        config_json = json.load(f)
    config = model_config(**config_json)

    # Initialize the model class using the loaded configuration
    model = model_cls(config)
    # Load the model weights from the checkpoint
    model.load_weights(checkpoint_dir)
    return model


def gen_mask(mask_ratio: float, pad_mask: torch.Tensor) -> torch.Tensor:
    """
    Generate a mask for use when pre-training. True represents values
    that are masked. Currently, this function is not very well optimized.
    However, because this function is expected to be called during data
    loading, it should be running in a non-blocking and multi-threaded
    setup on the cpu. Therefore, it should have no impact on the runtime
    unless the forward pass of on the GPU is very fast or the CPU is very
    slow.

    Parameters
    ----------
    mask_ratio : float
        Percentage of tokens to mask out
    pad_mask : torch.Tensor
        A boolean mask where positions in the input corresponding to
        padding have value True
    """
    if mask_ratio < 0 or mask_ratio > 1:
        raise ValueError(f"Mask ratio must be between 0 and 1, but was given "
                         f"{mask_ratio}")

    ratio = mask_ratio
    per_sample_n = (~pad_mask).sum(dim=1)
    n_masked_per_sample = (per_sample_n * ratio).ceil().int()
    mask = torch.zeros(pad_mask.shape, dtype=torch.bool, device=pad_mask.device)
    for i in range(pad_mask.shape[0]):
        idxs = random.sample(range(per_sample_n[i].item()), n_masked_per_sample[i].item())
        for j in idxs:
            mask[i, j] = True

    max_masked = n_masked_per_sample.max()
    diff_from_max = (n_masked_per_sample - max_masked)
    for i in range(diff_from_max.shape[0]):
        for j in range(pad_mask.shape[1] + diff_from_max[i], pad_mask.shape[1]):
            mask[i, j] = True

    return mask

from typing import Optional
import random
import json
from pathlib import Path
import warnings

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
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
                stacklevel=2,
            )
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


def gen_mask(mask_ratio: float, pad_mask: torch.Tensor, single: bool = False) -> torch.Tensor:
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
    if single:
        max_masked = torch.tensor(pad_mask.shape[1] * ratio).ceil().int()
    else:
        max_masked = n_masked_per_sample.max()
    diff_from_max = (n_masked_per_sample - max_masked)
    for i in range(diff_from_max.shape[0]):
        for j in range(pad_mask.shape[1] + diff_from_max[i], pad_mask.shape[1]):
            mask[i, j] = True

    return mask

# Credit timm and copyright 2020 Ross Wightman
# Implementation of stochastic depth from this paper:
# Deep Networks with Stochastic Depth (https://arxiv.org/abs/1603.09382)
# Originally taken from here:
# https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/drop.py
def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


def get_drop_path(drop_path_rate: float, layer_id: int, depth: int):
    """Calculate stochastic depth probability and return the initialized layer.
    """
    if drop_path_rate > 0.:
        if layer_id == 0:
            return DropPath(0.)
        else:
            return DropPath(
                (layer_id / (depth - 1)) * drop_path_rate)
    else:
        return nn.Identity()


def get_encoder_size(size: str):
    """
    Get the parameters of a specific RoMA model encoder size.
    """
    match size:
        case "RoMA-tiny":
            return {
                "d_model": 180,
                "nhead": 3,
                "depth": 12
            }
        case "RoMA-small":
            return {
                "d_model": 432,
                "nhead": 6,
                "depth": 12
            }
        case "RoMA-base":
            return {
                "d_model": 720,
                "nhead": 12,
                "depth": 12
            }
        case "RoMA-large":
            return {
                "d_model": 960,
                "nhead": 16,
                "depth": 24
            }
        case _:
            raise ValueError(f"Unknown encoder size: {size}")

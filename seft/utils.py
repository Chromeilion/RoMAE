from pydantic_core import PydanticUndefined
import torch
import torch.nn as nn
import math


def add_model(parser, model):
    """Add Pydantic model to an ArgumentParser
    Thanks miksus:
    https://stackoverflow.com/questions/72741663/argument-parser-from-a-pydantic-model
    """
    fields = model.__fields__
    for name, field in fields.items():
        parser.add_argument(
            f"--{name}",
            dest=name,
            type=field.annotation,
            default=PydanticUndefined,
            help=field.description,
        )


class CosineLRScheduleWithWarmup(torch.optim.lr_scheduler.LRScheduler):
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


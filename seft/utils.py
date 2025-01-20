from dataclasses import dataclass

from pydantic_settings import BaseSettings
from pydantic_core import PydanticUndefined
import torch
import torch.nn as nn

@dataclass
class TestModelConfig:
    criterion: nn.Module = nn.MSELoss()


class TestModel(nn.Module):
    """Very basic model with a single parameter to make sure everything works.
    Ideally should just find the mean of the data.
    """
    def __init__(self, config: TestModelConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.param = nn.Parameter(torch.zeros(1))

    def forward(self, label = None, *args, **kwargs):
        pred = self.param
        loss = None
        if label is not None:
            loss = self.config.criterion(pred, label)
        return pred, loss


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
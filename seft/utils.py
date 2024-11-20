from dataclasses import dataclass

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

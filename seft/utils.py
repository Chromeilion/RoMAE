from pydantic_core import PydanticUndefined
import torch


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


class CosineLRScheduleWithWarmup:
    def __init__(self, optimizer, warmup_steps, total_steps):
        if warmup_steps > total_steps:
            raise ValueError("Warmup steps must be less than total steps")
        self.cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps
        )
        self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-5,
            end_factor=1.,
            total_iters=warmup_steps,
        )
        self.current_step = 0

    def step(self):
        if self.current_step < self.warmup_scheduler.total_iters:
            self.warmup_scheduler.step()
        else:
            self.cosine_scheduler.step()
        self.current_step += 1

    def state_dict(self):
        return {"current_step": self.current_step}

    def load_state_dict(self, state_dict):
        self.current_step = state_dict["current_step"]

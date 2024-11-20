import os
import torch
from typing import Callable, Any
from dataclasses import dataclass, field
import tqdm
import torch._dynamo

torch._dynamo.config.suppress_errors = True
PathLike = str | os.PathLike

@dataclass
class TrainerConfig:
    optimizer: torch.optim.Optimizer
    base_lr: float
    epochs: int
    run_name: str
    eval_every: int
    save_every: int
    batch_size: int
    checkpoint_dir: PathLike = "./checkpoints"
    num_dataset_workers: int = 4
    collate_fn: Callable[[Any], Any] = torch.utils.data.dataloader.default_collate
    optimizer_args: dict[str, Any] = field(default_factory=dict)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class Trainer:
    """
    Trainer class. Not as flexible as the Huggingface one, but gets the job
    done.
    """
    def __init__(self, config: TrainerConfig):
        self.config: TrainerConfig = config

    def train(self, dataset: torch.utils.data.Dataset, model: torch.nn.Module):
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.config.batch_size, shuffle=True,
            num_workers=self.config.num_dataset_workers, pin_memory=True,
            collate_fn=self.config.collate_fn
        )
        optim = self.config.optimizer(
            model.parameters(), lr=self.config.base_lr, **self.config.optimizer_args
        )
        model = torch.compile(model, mode="reduce-overhead")
        model.train()
        model.to(self.config.device)
        step_counter = 0
        with tqdm.tqdm(total=len(dataloader)*self.config.epochs,
                       desc="step") as pbar:
            for epoch in range(self.config.epochs):
                for modelargs in dataloader:
                    modelargs = {key: val.to(self.config.device) for key, val in modelargs.items()}
                    optim.zero_grad()
                    # The model is expected to return its own loss
                    _, loss = model(**modelargs)
                    loss.backward()
                    optim.step()

                    if step_counter % self.config.save_every == 0:
                        self.save()
                    if step_counter % self.config.eval_every == 0:
                        self.eval()

                    step_counter += 1
                    pbar.update(1)

    def eval(self):
        ...

    def save(self):
        ...
    
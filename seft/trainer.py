from datetime import datetime
import os
import torch
from typing import Callable, Any
from dataclasses import dataclass, field
import tqdm
import torch._dynamo
import wandb
from torch.distributed import destroy_process_group

torch._dynamo.config.suppress_errors = True
PathLike = str | os.PathLike


@dataclass
class TrainerConfig:
    optimizer: torch.optim.Optimizer
    base_lr: float
    epochs: int
    eval_every: int
    save_every: int
    batch_size: int
    checkpoint_dir: PathLike = "./checkpoints"
    num_dataset_workers: int = 0
    run_name: str = datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
    project_name: str = "SEFT"
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
        wandb.init(project=config.project_name, name=config.run_name)
        wandb.config.base_lr = config.base_lr
        wandb.config.epochs = config.epochs
        wandb.config.save_every = config.save_every
        wandb.config.batch_size = config.batch_size
        wandb.config.optimizer_args = config.optimizer_args

    def train(self, train_dataset: torch.utils.data.Dataset,
              test_dataset: torch.utils.data.Dataset,
              model: torch.nn.Module):
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_dataset_workers,
            pin_memory=True,
            collate_fn=self.config.collate_fn
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_dataset_workers,
            pin_memory=True,
            collate_fn=self.config.collate_fn
        )
        optim = self.config.optimizer(
            model.parameters(), lr=self.config.base_lr, **self.config.optimizer_args
        )
#        model = torch.compile(model)
        model.train()
        model.to(self.config.device)
        step_counter = 0
        with tqdm.tqdm(total=len(train_dataloader)*self.config.epochs,
                       desc="Training") as pbar:
            for epoch in range(self.config.epochs):
                for modelargs in train_dataloader:
                    modelargs = {key: val.to(self.config.device) for key, val in modelargs.items()}
                    optim.zero_grad()
                    # The model is expected to return its own loss
                    _, loss = model(**modelargs)
                    loss.backward()
                    optim.step()

                    if step_counter % self.config.save_every == 0:
                        with torch.no_grad():
                            self.save(model, optim, step_counter)
                    if step_counter % self.config.eval_every == 0:
                        with torch.no_grad():
                            model.eval()
                            self.eval(model, loss, test_dataloader)
                            model.train()

                    step_counter += 1
                    pbar.update(1)

    def eval(self, model, loss_train, test_dataloader):
        wandb.log({"loss/train": loss_train.item()})
        print(f"Train loss: {loss_train.item()}\n")
        loss = 0
        for modelargs in tqdm.tqdm(test_dataloader, desc="Evaluating"):
            modelargs = {key: val.to(self.config.device) for key, val in
                         modelargs.items()}
            _, loss_ = model(**modelargs)
            loss += loss_ / len(test_dataloader)
        wandb.log({"loss/validation": loss.item()})
        print(f"Test loss: {loss.item()}\n")

    def save(self, model, optim, step_counter):
        ...
    
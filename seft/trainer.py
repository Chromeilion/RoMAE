from datetime import datetime
import os
from pathlib import Path
import torch
from typing import Callable, Any, Optional
import tqdm
import torch._dynamo
import wandb
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from accelerate.data_loader import skip_first_batches


torch._dynamo.config.suppress_errors = True
PathLike = str | os.PathLike

noop = lambda *_, **__: None

class TrainerConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix='SEFT_TRAINER_',
        env_file='.env',
        extra="ignore"
    )
    base_lr: float
    epochs: int
    eval_every: int
    save_every: int
    batch_size: int
    optimizer: str = Field("sgd", description="Either sgd or adamw")
    checkpoint_dir: str = Field(
        "./checkpoints",
        description="Where to save the model throughout training"
    )
    num_dataset_workers: int = Field(
        os.cpu_count()//2 - 1,
        description="Number of dataloader workers"
    )
    run_name: str = Field(
        datetime.today().strftime('%Y-%m-%d-%H-%M-%S'),
        description="Name of the run"
    )
    project_name: str = Field(
        "SEFT",
        description="Name of the project ind WandB"
    )
    collate_fn: Callable[[Any], Any] = Field(
        torch.utils.data.dataloader.default_collate,
        description="Custom collate function"
    )
    optimizer_args: Optional[dict[str, Any]] = Field(
        None,
        description="Arguments to be passed to the optimizer"
    )
    device: str = Field(
        "cuda" if torch.cuda.is_available() else "cpu",
        description="Device to use when training"
    )

class Trainer:
    """
    Trainer class. Not as flexible as the Huggingface one, but gets the job
    done.
    """
    def __init__(self, config: TrainerConfig):
        self.config: TrainerConfig = config
        self.evaluate_callback = noop

    def init_wandb(self, model):
        conf = {"trainer": dict(self.config), "model": dict(model.config)}
        wandb.init(project=self.config.project_name, name=self.config.run_name,
                   config=conf)

    def get_optim(self):
        match self.config.optimizer:
            case "sgd":
                optim_fn = torch.optim.SGD
            case "adamw":
                optim_fn = torch.optim.AdamW
            case _:
                raise AttributeError(
                    f"Invalid optimizer in config: {self.config.optimizer}"
                )
        return optim_fn

    def get_optimizer_args(self):
        if self.config.optimizer_args is not None:
            return self.config.optimizer_args
        return {}

    def train(self, train_dataset: torch.utils.data.Dataset,
              test_dataset: torch.utils.data.Dataset,
              model: torch.nn.Module,
              checkpoint: Optional[str] = None):
        self.init_wandb(model)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_dataset_workers,
            pin_memory=True,
            collate_fn=self.config.collate_fn
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_dataset_workers,
            pin_memory=True,
            collate_fn=self.config.collate_fn
        )
        optim = self.get_optim()(
            model.parameters(),
            lr=self.config.base_lr,
            **self.get_optimizer_args()
        )
        step_counter = 0
        current_epoch = 0
        if checkpoint is not None:
            step_counter, train_dataloader = self.load_from_checkpoint(
                checkpoint, model, optim, train_dataloader
            )
            current_epoch = step_counter // len(train_dataloader)

#        model = torch.compile(model, mode="reduce-overhead")
        model.train()
        model.to(self.config.device)
        with tqdm.tqdm(total=len(train_dataloader)*self.config.epochs,
                       desc="Training", initial=step_counter) as pbar:
            for epoch in range(current_epoch, self.config.epochs):
                for modelargs in train_dataloader:
                    modelargs = {key: val.to(self.config.device) for
                                 key, val in modelargs.items()}
                    optim.zero_grad()
                    # The model is expected to return its own loss
                    _, loss = model(**modelargs)
                    loss.backward()
                    optim.step()

                    if step_counter % self.config.save_every == 0:
                        with torch.no_grad():
                            self.save_checkpoint(model, optim, step_counter)
                    if step_counter % self.config.eval_every == 0:
                        with torch.no_grad():
                            model.eval()
                            self.evaluate(model, loss, test_dataloader)
                            model.train()

                    step_counter += 1
                    pbar.update(1)

    def evaluate(self, model, loss_train, test_dataloader):
        wandb.log({"loss/train": loss_train.item()})
        print(f"Train loss: {loss_train.item()}\n")
        loss = 0
        for modelargs in tqdm.tqdm(test_dataloader, desc="Evaluating"):
            modelargs = {key: val.to(self.config.device) for key, val in
                         modelargs.items()}
            _, loss_ = model(**modelargs)
            loss = loss + loss_ / len(test_dataloader)
        wandb.log({"loss/validation": loss.item()})
        self.evaluate_callback(model, loss_train, test_dataloader)

    def save_checkpoint(self, model, optim, step_counter):
        savedir = Path(self.config.checkpoint_dir)
        savedir.mkdir(exist_ok=True)

        torch.save(
            {"model_state_dict": model.state_dict(),
             "optimizer_state_dict": optim.state_dict(),
             "step_counter": step_counter},
            savedir/f"checkpoint-{step_counter}.tar"
        )

    @staticmethod
    def load_from_checkpoint(checkpoint, model, optim, dataloader):
        data = torch.load(checkpoint)
        model.load_state_dict(data["model_state_dict"])
        optim.load_state_dict(data["optimizer_state_dict"])
        step = data["step_counter"]
        n_epochs = step // len(dataloader)
        n_steps_in_current_epoch = step - (n_epochs * len(dataloader))
        dl = skip_first_batches(dataloader, n_steps_in_current_epoch)

        return step, dl

    def set_evaluate_callback(self, callback):
        self.evaluate_callback = callback

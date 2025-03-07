from datetime import datetime
import os
import random
from pathlib import Path
import numpy as np
import torch
import json
from typing import Any, Optional
import tqdm
from robite.utils import CosineLRScheduleWithWarmup
import torch._dynamo
import wandb
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from accelerate import Accelerator
from accelerate.data_loader import skip_first_batches


torch._dynamo.config.suppress_errors = True
PathLike = str | os.PathLike

noop = lambda *_, **__: None

class TrainerConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix='ROBITE_TRAINER_',
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
        os.cpu_count() - 1,
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
    optimizer_args: Optional[dict[str, Any]] = Field(
        None,
        description="Arguments to be passed to the optimizer"
    )
    device: str = Field(
        "cuda" if torch.cuda.is_available() else "cpu",
        description="Device to use when training"
    )
    warmup_steps: int = Field(
        2000,
        description="Number of warmup steps"
    )
    lr_schedule: str = Field(
        "cosine",
        description="What learning rate scheduler to use"
    )
    max_checkpoints: Optional[int] = Field(
        4,
        description="Maximum number of checkpoints to keep saved on disk"
    )
    gradient_clip: Optional[float] = Field(
        5.0,
        description="Gradient clipping value"
    )
    random_seed: int = Field(42)


class Trainer:
    """
    Trainer class. Not as flexible as the Huggingface one, but gets the job
    done.
    """
    def __init__(self, config: TrainerConfig):
        self.config: TrainerConfig = config
        self.evaluate_callback = noop
        self.post_train_hook = noop
        self.run = None

    def init_wandb(self, model):
        conf = {"trainer": dict(self.config), "model": dict(model.config)}
        self.run = wandb.init(
            project=self.config.project_name,
            name=self.config.run_name,
            config=conf
        )

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

    def get_lr_scheduler(self, optimizer, total_steps):
        if self.config.lr_schedule == "cosine":
            return CosineLRScheduleWithWarmup(
                optimizer, self.config.warmup_steps, total_steps
            )

    def get_optimizer_args(self):
        if self.config.optimizer_args is not None:
            return self.config.optimizer_args
        return {}

    def set_seeds(self):
        torch.manual_seed(self.config.random_seed)
        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)

    def train(self, train_dataset: torch.utils.data.Dataset,
              test_dataset: torch.utils.data.Dataset,
              model: torch.nn.Module,
              checkpoint: Optional[str] = None,
              train_collate_fn=torch.utils.data.dataloader.default_collate,
              eval_collate_fn=torch.utils.data.dataloader.default_collate):
        Path(self.config.checkpoint_dir).mkdir(exist_ok=True, parents=True)
        self.set_seeds()
        accelerator = Accelerator(project_dir=self.config.checkpoint_dir)
        self.init_wandb(model)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_dataset_workers,
            pin_memory=True,
            shuffle=True,
            collate_fn=train_collate_fn,
            prefetch_factor=2,
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_dataset_workers,
            pin_memory=True,
            collate_fn=eval_collate_fn,
            prefetch_factor=2
        )
        optim = self.get_optim()(
            model.parameters(),
            lr=self.config.base_lr,
            **self.get_optimizer_args()
        )
        scheduler = self.get_lr_scheduler(
            optim,
            len(train_dataloader) * self.config.epochs
        )
        model, optim, train_dataloader, scheduler = accelerator.prepare(
            model, optim, train_dataloader, scheduler,
            device_placement=[True, False, True, False]
        )

        step_counter = 0
        current_epoch = 0
        if checkpoint is not None:
            step_counter, current_epoch, train_dataloader = self.load_from_checkpoint(
                checkpoint, accelerator, train_dataloader
            )
        model.train()
        model.to(accelerator.device)

        with tqdm.tqdm(total=len(train_dataloader)*self.config.epochs,
                       desc="Training", initial=step_counter) as pbar:
            for epoch in range(current_epoch, self.config.epochs):
                for modelargs in train_dataloader:
                    optim.zero_grad()
                    # The model is expected to return its own loss
                    _, loss = model(**modelargs)
                    if loss is not None:
                        accelerator.backward(loss)
                    if self.config.gradient_clip is not None:
                        accelerator.clip_grad_norm_(
                            model.parameters(),
                            self.config.gradient_clip
                        )
                    optim.step()

                    if step_counter % self.config.save_every == 0 and accelerator.is_main_process and step_counter > 0:
                        with torch.no_grad():
                            self.save_checkpoint(accelerator, model, step_counter)
                    if step_counter % self.config.eval_every == 0 and accelerator.is_main_process and step_counter > 0:
                        with torch.no_grad():
                            model.eval()
                            self.evaluate(model, loss, test_dataloader, optim, step_counter)
                            model.train()

                    scheduler.step()
                    step_counter += 1
                    pbar.update(1)

        if accelerator.is_main_process:
            with torch.no_grad():
                model.eval()
                self.evaluate(model, loss, test_dataloader, optim,
                              step_counter)
                self.save_checkpoint(accelerator, model, step_counter)
                self.post_train_hook(model, self.run)
                model.train()
            self.run.finish()

    def post_train(self, *args, **kwargs):
        self.post_train_hook(*args, **kwargs)

    def set_post_train_hook(self, hook):
        self.post_train_hook = hook

    def evaluate(self, model, loss_train, test_dataloader, optim, step):
        grads = [
            param.grad.detach().flatten()
            for param in model.parameters()
            if param.grad is not None
        ]
        norm = torch.cat(grads).norm()
        self.run.log({"train/lr": optim.param_groups[0]["lr"]}, step=step)
        self.run.log({"train/gradient_norm": norm}, step=step)
        if loss_train is not None:
            self.run.log({"loss/train": loss_train.item()}, step=step)
            print(f"Train loss: {loss_train.item()}\n")
        loss = 0
        for modelargs in tqdm.tqdm(test_dataloader, desc="Evaluating"):
            modelargs = {key: val.to(model.device) for key, val in
                         modelargs.items()}
            _, loss_ = model(**modelargs)
            loss = loss + loss_ / len(test_dataloader)
        self.run.log({"loss/validation": loss.item()}, step=step)
        self.evaluate_callback(model, loss_train, test_dataloader)

    def save_checkpoint(self, accelerator: Accelerator, model, step_counter):
        savedir = Path(self.config.checkpoint_dir)/f"checkpoint-{step_counter}"
        savedir.mkdir(exist_ok=True)

        accelerator.save_state(str(savedir))

        with open(savedir/"model_config.json", "w") as f:
            json.dump(model.config.model_dump(), f, indent=4)

        with open(savedir/"trainer_config.json", "w") as f:
            json.dump(self.config.model_dump(), f, indent=4)

        with open(savedir/"trainer_state.json", "w") as f:
            json.dump({"step": step_counter}, f, indent=4)

        self.remove_old_checkpoints(self.config.checkpoint_dir)

    def remove_old_checkpoints(self, checkpoint_dir):
        if self.config.max_checkpoints is None:
            return

        all_checkpoints = os.listdir(checkpoint_dir)
        if len(all_checkpoints) > self.config.max_checkpoints:
            sorted_checkpoints = sorted(all_checkpoints, key=lambda x: int(x.split("-")[-1]))
            for i in range(0, len(all_checkpoints) - self.config.max_checkpoints):
                for j in os.listdir(Path(checkpoint_dir)/sorted_checkpoints[i]):
                    os.unlink(Path(checkpoint_dir)/sorted_checkpoints[i]/j)
                os.rmdir(Path(checkpoint_dir)/sorted_checkpoints[i])

    @staticmethod
    def load_from_checkpoint(checkpoint, accelerator, dataloader):
        accelerator.load_state(checkpoint)
        with open(Path(checkpoint)/"trainer_state.json", "r") as f:
            state = json.load(f)

        step = state["step"]
        current_epoch = step // len(dataloader)
        step_in_epoch = step - current_epoch * len(dataloader)
        dl = skip_first_batches(dataloader, step_in_epoch)

        return step, current_epoch, dl

    def set_evaluate_callback(self, callback):
        self.evaluate_callback = callback

import math
import os
import random
import json
from pathlib import Path
from typing import Any, Optional
from datetime import datetime

import numpy as np
import torch
import tqdm
import torch._dynamo
import wandb
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from accelerate import Accelerator
from accelerate.data_loader import skip_first_batches

from roma.utils import CosineLRScheduleWithWarmup

# Supress errors if torch compile is used on unsupported hardware
torch._dynamo.config.suppress_errors = True

PathLike = str | os.PathLike

# A useful no-op function
noop = lambda *_, **__: None

class TrainerConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix='ROMA_TRAINER_',
        env_file='.env',
        extra="ignore"
    )
    base_lr: float
    epochs: int
    eval_every: int
    save_every: int
    batch_size: int
    optimizer: str = Field("adamw", description="Either sgd or adamw")
    checkpoint_dir: str = Field(
        "./checkpoints",
        description="Where to save the model throughout training"
    )
    num_dataset_workers: int = Field(
        os.cpu_count() - 1,
        description="Number of dataloader workers to spawn"
    )
    run_name: str = Field(
        datetime.today().strftime('%Y-%m-%d-%H-%M-%S'),
        description="Name of the run. Defaults to the date"
    )
    project_name: str = Field(
        "RoMA",
        description="Name of the project in WandB"
    )
    entity_name: str = Field(
        "rmae",
        description="Name of the entity in WandB, should be the same as "
                    "the team you are on."
    )
    optimizer_args: dict[str, Any] = Field(
        {},
        description="Additional arguments to be passed to the optimizer"
    )
    warmup_steps: int = Field(
        2000,
        description="Number of warmup steps"
    )
    lr_schedule: str = Field(
        "cosine",
        description="What learning rate scheduler to use, currently only "
                    "supports cosine."
    )
    max_checkpoints: Optional[int] = Field(
        4,
        description="Maximum number of checkpoints to keep saved on disk "
                    "at one time"
    )
    gradient_clip: Optional[float] = Field(
        1.0,
        description="Gradient norm clipping value. Defaults to 1."
    )
    random_seed: int = Field(42)
    lr_scaling: bool = Field(
        False,
        description="Whether to scale the learning rate with the number of "
                    "processes. When enabled, the learning rate will be scaled "
                    "to a new value using base_lr*np, where np is the "
                    "number of processes being used for training."
    )


class Trainer:
    """
    Trainer class. Similar to the Huggingface Trainer but simpler.
    """
    def __init__(self, config: TrainerConfig):
        self.config: TrainerConfig = config
        self.evaluate_callback = noop
        self.post_train_hook = noop
        self.run = None

    def init_wandb(self, accelerator, model):
        """Initialize Weights and Biases if currently the main process.
        """
        if accelerator.is_main_process:
            conf = {"trainer": self.config.model_dump(),
                    "model": model.config.model_dump()}
            self.run = wandb.init(
                entity=self.config.entity_name,
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

    def get_lr(self, accelerator):
        if self.config.lr_scaling:
            return self.config.base_lr * accelerator.num_processes
        return self.config.base_lr

    def get_lr_scheduler(self, optimizer, warmup_steps, total_steps):
        if self.config.lr_schedule == "cosine":
            return CosineLRScheduleWithWarmup(
                optimizer, warmup_steps, total_steps
            )

    def set_seeds(self):
        torch.manual_seed(self.config.random_seed)
        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)

    def get_dataloaders(self, train_dataset, test_dataset,
                            train_collate_fn, eval_collate_fn):
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
        return train_dataloader, test_dataloader

    def train(self, train_dataset: torch.utils.data.Dataset,
              test_dataset: torch.utils.data.Dataset,
              model: torch.nn.Module,
              checkpoint: Optional[str] = None,
              train_collate_fn=torch.utils.data.dataloader.default_collate,
              eval_collate_fn=torch.utils.data.dataloader.default_collate):
        # Create the checkpoint directory if it doesn't exist
        Path(self.config.checkpoint_dir).mkdir(exist_ok=True, parents=True)
        self.set_seeds()
        accelerator = Accelerator(
            project_dir=self.config.checkpoint_dir,
            step_scheduler_with_optimizer=False
        )
        self.init_wandb(accelerator=accelerator, model=model)
        train_dataloader, test_dataloader = self.get_dataloaders(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            train_collate_fn=train_collate_fn,
            eval_collate_fn=eval_collate_fn
        )
        optim = self.get_optim()(
            model.parameters(),
            lr=self.get_lr(accelerator),
            **self.config.optimizer_args
        )
        scheduler = self.get_lr_scheduler(
            optimizer=optim,
            warmup_steps=self.config.warmup_steps,
            total_steps=len(train_dataloader) * self.config.epochs
        )
        model_config = model.config # Used when saving the model
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

        with tqdm.tqdm(total=len(train_dataloader)*self.config.epochs,
                       desc="Training", initial=step_counter) as pbar:
            for epoch in range(current_epoch, self.config.epochs):
                for modelargs in train_dataloader:
                    optim.zero_grad()
                    # The model is expected to return its own loss
                    _, loss = model(**modelargs)
                    if loss is not None:
                        accelerator.backward(loss)
                        if self.config.gradient_clip is not None and accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(
                                model.parameters(),
                                self.config.gradient_clip
                            )
                        optim.step()
                        scheduler.step()

                    if step_counter % self.config.save_every == 0 and accelerator.is_main_process and step_counter > 0:
                        with torch.no_grad():
                            self.save_checkpoint(accelerator, step_counter, model_config)
                    if step_counter % self.config.eval_every == 0 and accelerator.is_main_process and step_counter > 0:
                        with torch.no_grad():
                            model.eval()
                            self.evaluate(accelerator, model, loss, test_dataloader, optim, step_counter)
                            model.train()

                    step_counter += 1
                    pbar.update(1)

        if accelerator.is_main_process:
            with torch.no_grad():
                model.eval()
                self.evaluate(accelerator, model, loss, test_dataloader, optim,
                              step_counter)
                self.save_checkpoint(accelerator, step_counter, model_config)
                self.post_train_hook(model, self.run, accelerator.device)
                model.train()
            self.run.finish()
        accelerator.end_training()

    def set_post_train_hook(self, hook):
        """
        Set a function te be run at the end of training. The function
        should accept the model as the first argument and the wandb run
        as the second.
        """
        self.post_train_hook = hook

    def set_evaluate_callback(self, callback):
        """
        Set a function to be run right after model evaluation.
        Should accept the model, the training loss, and the test
        dataloader.
        """
        self.evaluate_callback = callback

    def evaluate(self, accelerator, model, loss_train, test_dataloader, optim, step):
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
            modelargs = {key: val.to(accelerator.device) for key, val in
                         modelargs.items()}
            _, loss_ = model(**modelargs)
            loss = loss + loss_ / len(test_dataloader)
        self.run.log({"loss/validation": loss.item()}, step=step)
        self.evaluate_callback(model, loss_train, test_dataloader)

    def save_checkpoint(self, accelerator: Accelerator, step_counter,
                        model_config):
        savedir = Path(self.config.checkpoint_dir)/f"checkpoint-{step_counter}"
        savedir.mkdir(exist_ok=True)

        accelerator.save_state(str(savedir))

        with open(savedir/"model_config.json", "w") as f:
            json.dump(model_config.model_dump(), f, indent=4)

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

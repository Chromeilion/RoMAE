from os import PathLike

PathLike = str | PathLike

class TrainerConfig:
    def __init__(self,
                 optimizer,
                 base_lr: float,
                 epochs: int,
                 criterion,
                 run_name: str,
                 dataset,
                 model,
                 eval_every: int,
                 save_every: int,
                 checkpoint_dir: PathLike,
                 *args, **kwargs):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.epochs = epochs
        self.criterion = criterion
        self.run_name = run_name
        self.dataset = dataset
        self.model = model
        self.eval_every = eval_every
        self.save_every = save_every
        self.checkpoint_dir = checkpoint_dir


class Trainer:
    """
    Trainer class. Not as flexible as the Huggingface one, but gets the job
    done.
    """
    def __init__(self, config):
        self.config = config

    def train(self):
        self.config.model.train()
        step_counter = 0
        for epoch in range(self.config.epochs):
            for modelargs, label in self.config.dataset:
                self.config.optimizer.no_grad()
                logits = self.config.model(**modelargs)
                loss = self.config.criterion(logits, label)
                loss.backwards()
                self.config.optimizer.step()

                if step_counter % self.config.save_every == 0:
                    self.save()
                if step_counter % self.config.eval_every == 0:
                    self.eval()

                step_counter += 1

    def eval(self):
        ...

    def save(self):
        ...
    
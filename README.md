from roma.model import RoMAForClassificationConfig

# Rotary Masked Autoencoder

General purpose implementation of the RoMA model.
This package is intended to make using RoMA as easy as possible by providing 
configuration and training utilities for the model which function on one or 
more GPU's and even machines.

## Configuration Guide

All configuration for the classes is stored in PydanticSettings objects.
Because of this, all configuration can easily be set from a .env file as 
well as from within your code.
Each configuration has its own environment prefix, making it possible to
store everything in a single .env file. The universal prefix is ```ROMA_```. 
Prefixes for the Trainer, RoMAForClassification, and RoMAForPreTraining are 
```TRAINER_```,```CLASSIFIER_```, ```PRETRAIN_``` respectively. To see all the 
settings that can be changed, have a look at the top quarter of 
[model.py](https://github.com/Chromeilion/RoMA/blob/main/roma/model.py).

In order to set the base learning rate to 0.5 in the Trainer and the mask 
ratio to 0.7 in RoMAForPreTraining, one can put the following in a .env file:

```bash
ROMA_TRAINER_BASE_LR=0.5
ROMA_PRETRAIN_MASK_RATIO=0.7
```

When setting 'submodel' attributes from an environment variable, two underscores 
are required. Because the Transformer Encoder configuration is a submodel, 
it therefore follows this rule. For example, to set the number of layers to 3,
embedding dimension to 342, number of heads to 8, and the stochastic depth rate 
to 0.3 in the RoMAForClassification model, you can put the following in a .env:

```bash
ROMA_CLASSIFIER_ENCODER_CONFIG__DEPTH=3
ROMA_CLASSIFIER_ENCODER_CONFIG__NHEAD=8
ROMA_CLASSIFIER_ENCODER_CONFIG__D_MODEL=342
ROMA_CLASSIFIER_ENCODER_CONFIG__DROP_PATH_RATE=0.3
```

Lastly, when utilizing a pretrained weight to initialize RoMAForClassification,
many parameters such as the depth or width of the Transformer Encoder are 
automatically loaded from the pretrained checkpoint.
Parameters you might want to change such as dropout are not brought over, and 
will either reset to 0 or will take the value present in the environment.

## Training RoMA

First import everything:

```Python
from roma.model import RoMAForClassification, RoMAForClassificationConfig
from roma.trainer import Trainer, TrainerConfig
```

The RoMA model accepts 4 inputs:

- values: actual values going through the model (pixels, flux, etc.)
- positions: a 2D tensor of shape (3, n_tokens) which stores the 3D position of 
  each token after it has been converted to tubelets
- pad_mask: optional padding mask which marks what values should be ignored 
  during attention
- label: optional label which is used to calculate the loss

If the model receives a label, it will return a loss as well as the output 
logits. Assuming we have some PyTorch dataset that outputs a dictionary with 
the fields mentioned above, we can then conduct training as follows:

```Python
# we assume the datasets already exist
train_ds, test_ds = get_datasets()
trainer = Trainer(TrainerConfig())
model = RoMAForClassification(RoMAForClassificationConfig)
trainer.train(
  train_dataset=train_ds,
  test_dataset=test_ds,
  model=model
)
```

This will automatically train RoMAForClassification on the provided dataset, 
log all results to Weights and Biases, and save checkpoints.
If you have some custom evaluation code or a post-training hook you wish to run,
these can be set using ```set_evaluate_callback``` and ```set_post_train_hook```.
To see everything the trainer can do, I recommend you check out the actual 
code in [trainer.py](https://github.com/Chromeilion/RoMA/blob/main/roma/trainer.py) 
and have a look at the trainer config.
Overall, the process is very similar to the way HuggingFace works.
Pre-training works the exact same way and even accepts the same inputs (despite 
having no need for labels).

## Accelerate Support

The Trainer makes use of the [Accelerate](https://huggingface.co/docs/accelerate/en/index) 
package to enable many advanced features such as multi-GPU and multi-node
training, torch compile, and mixed precision. 
To use these features, simply make sure to run the trainer using the 
```accelerate launch``` command.

This repo also provides a useful [bash script](https://github.com/Chromeilion/RoMA/blob/main/scripts/run_experiment.sh) 
for use with SLURM on the Leonardo supercomputer.
To run the script, some environment variables have to be set in a .env file.
These are described at the top of the script.
With this, the script can wrap any executable Python module that makes use of 
the Trainer, forwarding all arguments to the module.

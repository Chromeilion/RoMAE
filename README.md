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
settings that can be changed, have a look at the top quarter of model.py.
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
model = RoMAForClassification(RoMAForClassificationConfig())
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

## Hyperparameter Tips

Here I'll list some tips for model sizes and hyperparameter values that may 
be useful to  try.

**Model Sizes:**

The most similar model to RoMA is the Vision Transformer. Therefore, it makes
sense to keep sizes similar. One difference between the two is that 
the head embedding dimension must be divisible by 6 in order to be able to 
apply the rotary positional embeddings. Therefore, the sizes I provide here 
are slightly different but still close to their ViT counterparts.

| Size       | d_model | nhead | depth |
|------------|---------|-------|-------|
| RoMA-tiny  | 180     | 3     | 12    |
| RoMA-small | 432     | 6     | 12    |
| RoMA-base  | 720     | 12    | 12    |
| RoMA-large | 960     | 16    | 24    |

To get a useful dictionary of all these parameters, you can use:

```Python
encoder_size_args = roma.utils.get_encoder_size("RoMA-tiny")
encoder_config = roma.model.EncoderConfig(**encoder_size_args)
```

**Regularization:**

RoMA implements both regular [dropout](https://arxiv.org/abs/1207.0580) and 
[stochastic depth](https://arxiv.org/abs/1603.09382). When using 
regularization, I recommend you take a look at ["How to train your ViT?"](https://arxiv.org/abs/2106.10270).
In general, if you add regularization you should also increase the number of 
epochs you are training for.
Data augmentation is very important too. This should be considered on a 
per-dataset basis however.

The encoder config provides fine-grained controls for where dropout is 
applied. The common approach is to use the same dropout value for:

- ```hidden_drop_rate``` (applies dropout in the MLP layer)
- ```attn_drop_rate``` (applies dropout on the attention probabilities)

The others can be set to zero, but feel free to experiment!

**Learning rate scaling:**

The Trainer has support for the [linear scaling rule](https://arxiv.org/abs/1706.02677). 
E.g. automatically scaling the learning rate with the number of processes 
using the following formula:

$$
lr_{new} = lr*np
$$

Where $np$ is the number of processes (with each process corresponding to a GPU). 
When training on one GPU, this does nothing, and even when training on a small number it probably does not 
matter. However, if you want to train on a larger number you should 
consider using it. This way, you can choose your hyperparameters by testing on 
one GPU, and then run the full training on many GPU's without having to change 
them.

## Interpolation Predictions

If you'd like to use RoMAForPreTraining for interpolation, you can first do a 
normal training run.
Then to use the learned weights, you can utilize the predict class method.
This will take in your known values plus a set of positions you wish to predict 
the values for and pass it through the model for you.
Because the method is batched, it also has support for padding.
When using padding, you must pass two padding arrays. One for the values, and 
one for the predictions.

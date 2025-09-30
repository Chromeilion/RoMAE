# Rotary Masked Autoencoder

Welcome to the implementation of the [RoMAE](https://arxiv.org/abs/2505.20535) 
model. This package is intended to make using RoMAE as easy as possible 
by providing the implementation of the model, configuration, and 
training utilities. To get started, install the package using pip from 
the repository root:

```bash
pip install .
```

We provide two model classes; ```romae.model.RoMAEForClassification``` 
and ```romae.model.RoMAEForPreTraining```. The provided trainer class 
```romae.trainer.Trainer``` can be used on both. If you're only 
interested in our transformer encoder implementation, feel free to use
```romae.model.Encoder```.
Our n-dimensional p-RoPE implementation can also be found in 
```romae.positional_embeddings.NDPRope```.

## Configuration Guide

All configuration for the classes is stored in PydanticSettings objects.
Because of this, configuration can easily be set from a .env file as 
well as from within your code.
Each configuration has its own environment prefix, making it possible to
store everything in a single .env file. The universal prefix is ```ROMAE_```. 
Prefixes for the Trainer, RoMAEForClassification, and RoMAEForPreTraining are 
```TRAINER_```,```CLASSIFIER_```, ```PRETRAIN_``` respectively. To see all the 
settings that can be changed, have a look at the top quarter of ```model.py```.
As an example, in order to set the base learning rate to 0.5 in the 
Trainer and the mask ratio to 0.7 in RoMAEForPreTraining, one can put 
the following in a .env file:

```bash
ROMAE_TRAINER_BASE_LR=0.5
ROMAE_PRETRAIN_MASK_RATIO=0.7
```

When setting 'submodel' attributes from an environment variable, two underscores 
are required. Because the Transformer Encoder configuration is a submodel, 
it therefore follows this rule. For example, to set the number of layers to 3,
embedding dimension to 342, number of heads to 8, and the stochastic depth rate 
to 0.3 in the RoMAEForClassification model, you can put the following in a .env:

```bash
ROMAE_CLASSIFIER_ENCODER_CONFIG__DEPTH=3
ROMAE_CLASSIFIER_ENCODER_CONFIG__NHEAD=8
ROMAE_CLASSIFIER_ENCODER_CONFIG__D_MODEL=342
ROMAE_CLASSIFIER_ENCODER_CONFIG__DROP_PATH_RATE=0.3
```

Lastly, when utilizing a pretrained weight to initialize RoMAEForClassification,
many parameters such as the depth or width of the Transformer Encoder are 
automatically loaded from the pretrained checkpoint.
Parameters you might want to change such as dropout are not brought over, and 
will either reset to 0 or will take the value present in the environment.

## Training RoMAE

First import everything:

```Python
from romae.model import RoMAEForClassification, RoMAEForClassificationConfig
from romae.trainer import Trainer, TrainerConfig
```

The RoMAE model accepts 4 inputs:

- values: actual values going through the model (pixels, flux, etc.),  
  must have 5 dimensions (batch, time, channel, height, width). If you're 
  working in less dimensions just set the extra dims to size 1
- positions: an ND tensor of shape (n_positional_dims, n_tokens) which 
  stores the N-dimensional position of each token
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
model = RoMAEForClassification(RoMAEForClassificationConfig())
trainer.train(
  train_dataset=train_ds,
  test_dataset=test_ds,
  model=model
)
```

This will automatically train RoMAEForClassification on the provided dataset, 
log all results to Weights and Biases, and save checkpoints.
If you have some custom evaluation code or a post-training hook you wish to run,
these can be set using ```set_evaluate_callback``` and ```set_post_train_hook```.
To see everything the trainer can do, I recommend you check out the actual 
code in ```trainer.py``` and have a look at the trainer config.
Overall, the process is very similar to the way HuggingFace works.
Pre-training works the exact same way and even accepts the same inputs (despite 
having no need for labels).

## Accelerate Support

The Trainer makes use of the [Accelerate](https://huggingface.co/docs/accelerate/en/index) 
package to enable many advanced features such as multi-GPU and multi-node
training, torch compile, and mixed precision. 
To use these features, simply make sure to run your training script 
using the ```accelerate launch``` command.

## Hyperparameter Tips

Here I'll list some tips for model sizes and hyperparameter values that may 
be useful to  try.

**Model Sizes:**

The most similar model to RoMAE is the Vision Transformer. Therefore, it makes
sense to keep sizes similar. One difference between the two is that 
the head embedding dimension must be divisible by 6 in order to be able to 
apply the rotary positional embeddings. Therefore, the sizes we provide here 
are slightly different but still close to their ViT counterparts.

| Size         | d_model | nhead | depth |
|--------------|---------|-------|-------|
| RoMAE-tiny   | 180     | 3     | 12    |
| RoMAE-small  | 432     | 6     | 12    |
| RoMAE-base   | 720     | 12    | 12    |

To get a useful dictionary of all these parameters, you can use:

```Python
import romae.utils
import romae.model

encoder_size_args = romae.utils.get_encoder_size("RoMAE-tiny")
encoder_config = romae.model.EncoderConfig(**encoder_size_args)
```

**Regularization:**

RoMAE implements both regular [dropout](https://arxiv.org/abs/1207.0580) and 
[stochastic depth](https://arxiv.org/abs/1603.09382). When using 
regularization, I recommend you take a look at ["How to train your ViT?"](https://arxiv.org/abs/2106.10270).
In general, if you add regularization you should also increase the number of 
epochs you are training for. The encoder config provides fine-grained 
controls for where dropout is applied. The common approach is to use 
the same dropout value for:

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
When training on one GPU, this does nothing. If you want to train on a 
larger number of GPU's however, you should consider using it. This way, 
you can choose your hyperparameters by testing on one GPU, and then run 
the full training on many GPU's without having to change them.

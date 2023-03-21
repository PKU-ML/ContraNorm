# Training ViT with ContraNorm

## Introduction
Our code is mostly based on the codebase of [DeiT](https://github.com/facebookresearch/deit) and the ``timm`` toolkit. You can refer to <a href="README_deit.md">README_deit.md</a> for details on the installation and running of the code. One notable difference is that in order to achieve a drop-in replacement of the normalzation layer, we rely on a newer version of ``timm=0.6.7`` than the default. You can easily install it with the zip file provided in this repo:

```pip install timm-v0.6.7.zip```


## ContraNorm

The ``ContraNorm`` module is implemented in ``models.py``. It is designed to be a direct drop-in replacement of the original LayerNorm, though some hyperparameters needed to be set for it to work as expected. The implementation of this module may be of independent interest for other architectures.

We include a flexible choice of hyperparameters of ContraNorm, and you may follow the experimental settings in the paper to find the suitable hyperparameters to reproduce the results.

## Examples

To train a ViT with ContraNorm, we can use the following script from ``README_deit.md`` by adding some configurations for ContraNorm.


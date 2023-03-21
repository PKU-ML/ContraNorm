# Training GNN with ContraNorm

Official pytorch code for ICLR 2023 paper [ContraNorm: A Contrastive Learning Perspective on Oversmoothing and Beyond](https://openreview.net/forum?id=SM7XkJouWHm)  

## Introduction

Our code is based on the implementation of [PairNorm](https://github.com/LingxiaoShawn/PairNorm). This code requires `pytorch_geometric>=1.3.2`. We mainly modify `NormLayer` in [layers.py](https://github.com/PKU-ML/ContraNorm/GNN/layers.py). It is designed to be a direct drop-in replacement of the original PairNorm, though some hyperparameters needed to be set for it to work as expected. The implementation of this module may be of independent interest for other architectures besides `DeepGCN` in [models.py](https://github.com/PKU-ML/ContraNorm/GNN/models.py).

## ContraNorm

For multi-layer GCN, we plug ContraNorm following the convolution operation in every layer, and tune the scale in {0.2, 0.5, 0.8, 1.0}. Empirically, a higher scale is preferred in a deeper layer setting. 

## Examples

For usage, you can type the following command in the terminal for different settings.

``` bash
python main.py --data cora --model DeepGCN --nlayer 16 --norm_mode CN --norm_scale 1.0 --use_layer_norm --hid 32 --epochs 200
```

You can also use the script to tune the hyperparameters.

``` bash
bash run_different_baselines.sh 0
```

Here, the single number denotes the index of GPU.

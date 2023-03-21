# ContraNorm

Official code for ICLR 2023 paper [ContraNorm: A Contrastive Learning Perspective on Oversmoothing and Beyond](https://openreview.net/forum?id=SM7XkJouWHm)

## Introduction

Oversmoothing is a common phenomenon in a wide range of GNNs and Transformers. Instead of characterizing oversmoothing from the view of complete collapse, we dive into a more general perspective of dimensional collapse in which representations lie in a narrow cone. Accordingly, inspired by the effectiveness of contrastive learning in preventing dimensional collapse, we propose a novel normalization layer called `ContraNorm`. Both theoretically and empirically, we demonstrate the effectiveness of our proposed ContraNorm on  alleviating collapse.
<div align="center">    
    <img src=".pics/contranorm.png" width = "300" height = "200" alt="ContraNorm" align=center />
</div>

We evaluate ContraNorm on 1) GLUE tasks with BERT and ALBERT as backbones; 2) ImageNet benchmarks with ViT as backbone; 3) node classification tasks with multi-layer GCN as backbone. In a nutshell, we achieve an average of 83.54 score on validation set of GLUE tasks compared to 82.59 with the vanilla BERT-base. On ImageNet100, a 24-layer DeiT with ContraNorm reaches 81.28% test accuracy compared to 76.76% with vanilla DeiT with the same layer setting. For experiments on graphs, GCN with ContraNorm also performs the best in deep layer settings compared to baselines such as PairNorm.

## File Structures

We organize our code in the following strucute. The detailed guidance is included in the README.md of each subfile (`BERT_gleu`, `GNN` and `ViT_imagenet`).

``` bash
ContraNorm/
├── README.md
├── BERT_gleu/
│   ├── README.md
│   ├── README_transformers.md
│   ├── run_glue_baselines.sh
│   ├── run_glue_baselines_al.sh
│   ├── run_different_baselines.sh 
│   ├── run_different_baselines_al.sh
│   ├── run_glue_no_trainer.py
│   └── transformers/
│       ├── models/
│       │   ├── bert/
│       │   │   ├── configuration_bert.py
│       │   │   ├── modeling_bert.py
│       │   │   └── ...
│       │   └── ...
│       └── ...
├── GNN/
│   ├── README.md
│   ├── data.py
│   ├── layers.py
│   ├── main.py
│   ├── models.py
│   ├── utils.py
│   ├── run_baselines.sh
│   ├── run_different_baselines.sh
│   ├── data/
│   └── ...
├── ViT_imagenet/
│   ├── README.md
│   ├── README_deit.md
│   ├── models.py
│   ├── models_v2.py
│   ├── main.py
│   ├── datasets.py
│   ├── losses.py
│   ├── run.sh
│   └── ...
└── ...
```

## Citation

If you use our code, please cite

```
@article{guo2023contranorm,
  title={ContraNorm: A Contrastive Learning Perspective on Oversmoothing and Beyond},
  author={Guo, Xiaojun and Wang, Yifei and Du, Tianqi and Wang, Yisen},
  journal={arXiv preprint arXiv:2303.06562},
  year={2023}
}
```

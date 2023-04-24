# GIANT-XRT+RevGAT

This is the repository for reproducing the results in our paper: [[ContraNorm: A Contrastive Learning Perspective on Oversmoothing and Beyond]](https://arxiv.org/abs/2303.06562.pdf) for the D-ContraNorm on ogbn-arxiv benchmark.

## Step 0: Install GIANT and get GIANT-XRT node features.
Please follow the instruction in [[GIANT]](https://github.com/amzn/pecos/tree/mainline/examples/giant-xrt) to get the GIANT-XRT node features. Note that if you generate your own pretrained node features from GIANT-XRT, you should be aware of your save path and modify the --pretrain_path (below in Step 3) accordingly.

## Step 1: Git clone this repo.
After following the steps in [[GIANT]](https://github.com/amzn/pecos/tree/mainline/examples/giant-xrt), go to the folder
`pecos/examples/giant-xrt`

Then git clone this repo in the folder `giant-xrt` directly.

## Step 2: Install additional packages.
If you install and run GIANT correctly, you should only need to additionally install [dgl>=0.5.3](https://github.com/dmlc/dgl). See [here](https://www.dgl.ai/pages/start.html) for pip/conda installation instruction for dgl.

## Step 3: Run the experiment.
Go to the folder `OGB`.

For reproducing our results for ogbn-arxiv dataset with GIANT-XRT features, run
`bash run_teacher.sh $gpu_id 2.0 2.0`, then
`bash run_student.sh $gpu_id 2.0 5.0`

```
New arguments
--dcn_use_scale design settings for D-ContraNorm
--dcn_scale $SCALE the scale of D-ContraNorm
--tau $TAU the tau value for D-ContraNorm
--gpu $GPU the index of gpu
--data_root_dir: path to save ogb datasets.
--pretrain_path: path to load GIANT-XRT features. Set it to 'None' for using ogb default features.
``` 

## Results
If execute correctly, you should have the following performance (using pretrained GIANT-XRT features).

|  | GIANT + RevGAT + KD | GIANT + RevGAT + KD + D-ContraNorm |
|---|---|---|
| Average val accuracy (%) | 77.16 ± 0.09  |  |
| Average test accuracy (%) | 76.15 ± 0.10 |  |

Number of params: 1304912

**Remark:** We do not carefully fine-tune RevGAT for our GIANT-XRT. It is possible to achieve higher performance by fine-tune it more carefully. For more details about RevGAT, please check the original README.

## Citation
If you find our code useful, please cite our ContraNorm paper and the GIANT and RevGAT references provided in the original repo.

Our ContraNorm paper:
```
@inproceedings{guo2023contranorm,
  title={ContraNorm: A Contrastive Learning Perspective on Oversmoothing and Beyond},
  author={Guo, Xiaojun and Wang, Yifei and Du, Tianqi and Wang, Yisen},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2023}
}

```
GIANT paper:
```
@inproceedings{chien2021node,
  title={Node Feature Extraction by Self-Supervised Multi-scale Neighborhood Prediction},
  author={Chien, Eli and Chang, Wei-Cheng and Hsieh, Cho-Jui and Yu, Hsiang-Fu and Zhang, Jiong and Milenkovic, Olgica and Dhillon, Inderjit S},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2022}
}
```

RevGAT paper:

```
@InProceedings{li2021gnn1000,
    title={Training Graph Neural Networks with 1000 layers},
    author={Guohao Li and Matthias Müller and Bernard Ghanem and Vladlen Koltun},
    booktitle={International Conference on Machine Learning (ICML)},
    year={2021}
}
```


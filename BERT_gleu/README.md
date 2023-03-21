# Training BERT with ContraNorm

Official code for ICLR 2023 paper [ContraNorm: A Contrastive Learning Perspective on Oversmoothing and Beyond](https://openreview.net/forum?id=SM7XkJouWHm)

## Introduction 

Our code is mostly based on the codebase of [transformers](https://github.com/huggingface/transformers) produced by Hugging-face. You can refer to [README_transformers.md](https://github.com/PKU-ML/ContraNorm/BERT_glue/README_transformers.md) for details on the installation and running of the code. Note that we have included the whole `transformers` source package in our code for the need of modifying backbone models. So there is no need to install the `transformers` package by pip or conda.

## ContraNorm

In detials, we modify the [modeling_bert.py](https://github.com/PKU-ML/ContraNorm/BERT_gleu/transformers/models/bert/modeling_bert.py) and [configuration_bert.py](https://github.com/PKU-ML/ContraNorm/BERT_gleu/transformers/models/bert/configuration_bert.py) for plugging in our ContraNorm layer. We provide multiple plugging positions of ContraNorm. You can assign one position simply with the hyperparameter `--pos`. The postion is encoded in integers (1, 2, 3, 4, 5, 6, 7) and the default setting reported in our paper is `--pos 7`.

## Examples

To finetune BERT with ContraNorm on GLEU benchmarks, you can run [run_glue_no_trainer.py](https://github.com/PKU-ML/ContraNorm/BERT_gleu/run_glue_no_trainer.py). The hyper-parameter relevant to ContraNorm is `--pos` (plugging position) and `--scale` (scaling factor $s$).

``` bash
CUDA_VISIBLE_DEVICES='0' python run_glue_no_trainer.py \
  --model_name_or_path bert-base-cased \
  --task_name rte \
  --max_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 5 \
  --output_dir output/rte \
  --pos 7 \
  --scale 0.005
```

We also provide bash scripts [run_glue_baselines.sh](https://github.com/PKU-ML/ContraNorm/BERT_gleu/run_glue_baselines.sh) for different normalization scales and plugging positions:

``` bash
bash run_glue_baselines.sh $TASK_NAME $SCALE $POS $DEVICE
```

For example, you can run BERT with ContraNorm of 0.01 scale, 7th position on 1th GPU, and evaluate it on COLA task by

``` bash
bash run_glue_baselines.sh cola 0.01 7 1
```

For tuning hyperparameters, you can edit and run [run_different_baselines.sh](https://github.com/PKU-ML/ContraNorm/BERT_gleu/run_different_baselines.sh):

```bash 
bash run_different_baselines.sh $DEVICE
```

We also provide implementation for the backbone `AlBERT` with ContraNorm. Likewise, you can run [run_glue_baselines_al.sh](https://github.com/PKU-ML/ContraNorm/BERT_gleu/run_glue_baselines_al.sh) and [run_different_baselines_al.sh](https://github.com/PKU-ML/ContraNorm/BERT_gleu/run_different_baselines_al.sh) for verification.

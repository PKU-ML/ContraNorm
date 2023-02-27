# Text classification examples
# ContraNorm 
Official pytorch source code for ContraNorm [paper](https://openreview.net/pdf?id=SM7XkJouWHm) (ICLR 2023). The code is based on [transformers library](https://github.com/huggingface/transformers) builed by hugging-face.

## GLUE tasks

**BERT base**

Based on the script [`run_glue_no_trainer.py`](https://github.com/PKU-ML/ContraNorm/blob/main/bert/run_glue_no_trainer.py).

You can run Bert with ContraNorm of different normalization scales and plugging position using [`run_glue_baselines.sh`](https://github.com/PKU-ML/ContraNorm/blob/main/bert/run_glue_baselines.sh)

```
bash run_glue_baselines.sh $TASK_NAME $SCALE $POS $DEVICE
```
For example, you can run Bert with ContraNorm of 0.005 scale, 7th position on 1th GPU, and evaluate it on RTE task like
```
bash run_glue_baselines.sh rte 0.005 7 1
```
The postions are encoded in integers and the choice is in {1, 2, 3, 4, 5, 6, 7}. The default setting is 7.


For tuning the hyperparameters, you can edit and run the script [`run_different_baselines.sh`](https://github.com/PKU-ML/ContraNorm/blob/main/bert/run_different_baselines.sh).
```
bash run_different_baselines.sh 0
```
The single number in the command is the index of GPU.


**ALBERT base**

Likewise, you can run AlBert with ContraNorm on GLUE tasks with [`run_glue_baselines_al.sh`](https://github.com/PKU-ML/ContraNorm/blob/main/bert/run_glue_baselines_al.sh) and [`run_different_baselines_al.sh`](https://github.com/PKU-ML/ContraNorm/blob/main/bert/run_different_baselines_al.sh)


## citation
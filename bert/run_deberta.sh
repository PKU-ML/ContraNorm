export TASK_NAME=$1
export EPOCHS=$2
export WARMS=$3
export BATCH=$4
export LR=$5
export SCALE=$6
export POS=$7
export DEVICE=$8
export model_name="microsoft/deberta-base"

CUDA_VISIBLE_DEVICES="$DEVICE" python run_glue_no_trainer.py \
  --model_name_or_path $model_name \
  --task_name $TASK_NAME \
  --dataset_name $TASK_NAME \
  --num_train_epochs $EPOCHS \
  --num_warmup_steps $WARMS \
  --per_device_train_batch_size $BATCH \
  --learning_rate $LR \
  --scale $SCALE \
  --pos $POS \
  --max_length 128 \
  --output_dir /data0/xjguo/ContraNorm/output/$model_name/$TASK_NAME \
  --seed 2021 \
  --num_hidden_layers 12 \


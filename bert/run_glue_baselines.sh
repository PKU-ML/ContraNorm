export TASK_NAME=$1
export SCALE=$2
export POS=$3
export LR=$4
export DEVICE=$5
export model_name="bert-base-uncased"

CUDA_VISIBLE_DEVICES="$DEVICE" python run_glue_no_trainer.py \
  --model_name_or_path $model_name \
  --task_name $TASK_NAME \
  --dataset_name $TASK_NAME \
  --num_train_epochs 3 \
  --max_length 128 \
  --per_device_train_batch_size 32 \
  --output_dir /data0/xjguo/ContraNorm/output/$model_name/$TASK_NAME \
  --seed 2021 \
  --scale $SCALE \
  --pos $POS \
  --num_hidden_layers 12 \
  --learning_rate $LR 


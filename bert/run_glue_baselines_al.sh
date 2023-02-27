export TASK_NAME=$1
export SCALE=$2
export POS=$3
export LAYER=$4
export EPOCH=$5
export DEVICE=$6
export model_name="albert-base-v2"

CUDA_VISIBLE_DEVICES="$DEVICE" python run_glue_no_trainer.py \
  --model_name_or_path $model_name \
  --task_name $TASK_NAME \
  --dataset_name $TASK_NAME \
  --learning_rate 2e-5 \
  --num_train_epochs $EPOCH \
  --max_length 128 \
  --per_device_train_batch_size 32 \
  --output_dir /data0/xjguo/ContraNorm/output/$model_name/$TASK_NAME \
  --seed 2021 \
  --scale $SCALE \
  --pos $POS \
  --num_hidden_layers $LAYER \

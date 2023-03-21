export TASK_NAME=$1
export SCALE=$2
export POS=$3
export DEVICE=$4
export model_name="bert-base-uncased"

CUDA_VISIBLE_DEVICES="$DEVICE" python run_glue_no_trainer.py \
  --model_name_or_path $model_name \
  --task_name $TASK_NAME \
  --scale $SCALE \
  --pos $POS \
  --num_train_epochs 5 \
  --max_length 128 \
  --per_device_train_batch_size 32 \
  --output_dir output/$model_name/$TASK_NAME \
  --seed 2021 \
  --num_hidden_layers 12 \
  --learning_rate 2e-5 


export TASK_NAME=qnli

python run_glue.py \
  --model_name_or_path distilbert-base-uncased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir /tmp/$TASK_NAME/ \
  --seed 2021 \
  --vis_step 50 \
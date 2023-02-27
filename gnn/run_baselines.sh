export DATA=$1
export MODE=$2
export SCALE=$3
export HID=$4
export LAYER=$5
export EPOCH=$6
export DEVICE=$7
export model_name="DeepGCN"

CUDA_VISIBLE_DEVICES="$DEVICE" python main.py \
  --data $DATA \
  --norm_mode $MODE \
  --norm_scale $SCALE \
  --nlayer $LAYER \
  --epochs $EPOCH \
  --model $model_name \
  --hid $HID \
  --use_layer_norm

# install new timm first --master_port=2222
# cd /cache/deit 
# pip install --upgrade torch torchvision
# pip install timm-v0.6.7.zip
CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.launch --nproc_per_node=2 --use_env --master_port 10002 main.py --model deit_tiny_patch16_224 --batch-size 256 --data-path $1 --output_dir $2/deit_tiny_patch16_224/split_attn --split-attn & \
CUDA_VISIBLE_DEVICES=7,3 python -m torch.distributed.launch --nproc_per_node=2 --use_env --master_port 10001 main.py --model deit_tiny_patch16_224 --batch-size 256 --data-path $1 --output_dir $2/deit_tiny_patch16_224/vanilla
# CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --model deit_tiny_patch16_224 --batch-size 256 --data-path $1 --output_dir $2/deit_tiny_patch16_224/cn_scale1e-2_pos1 --scale 1e-2 --pos 1
# CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --model deit_tiny_patch16_224 --batch-size 256 --data-path $1 --output_dir $2/deit_tiny_patch16_224/cn_scale1e-2_pos2 --scale 1e-2 --pos 2
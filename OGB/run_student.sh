export GPU=$1

python ./examples/ogb_eff/ogbn_arxiv_dgl/main.py --data_root_dir ../datasets \
--pretrain_path proc_data_xrt/ogbn-arxiv/X.all.xrt-emb.npy \
--use-norm --use-labels --n-label-iters=1 --no-attn-dst --edge-drop=0.3 --input-drop=0.25 --n-layers 2 --dropout 0.75 --n-hidden 256 --save kd --backbone rev --group 2 --alpha 0.95 --temp 0.7 --mode student \
--n-runs 3 --n-epochs 1000 --dcn_scale 1.0 --tau 1.0 --dcn_use_scale \
--gpu $GPU
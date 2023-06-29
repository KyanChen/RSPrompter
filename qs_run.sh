#!/bin/bash
source ~/.bashrc
conda activate torch2mmcv2  # torch1mmcv1 torch1mmcv2 torch2mmcv1 torch2mmcv2
pip install albumentations
pip install importlib_metadata
pip install --upgrade mmengine
pip install instaboostfast
#pip install deepspeed
# pip install anypackage
# yum install which
# source /opt/rh/devtoolset-9/enable
# mim install mmcv>=2.0.0rc4

cd /mnt/search01/usr/chenkeyan/codes/lightning_framework
#TORCH_DISTRIBUTED_DEBUG=DETAIL
case $# in
0)
    python tools/train.py
    ;;
1)
    python tools/train.py --config $1
    ;;
2)
    python tools/train.py --config $1 --ckpt-path $2
    ;;
esac
# TORCH_DISTRIBUTED_DEBUG=DETAIL
#python train.py
#python -m torch.distributed.launch --nproc_per_node=$GPU_NUM --nnodes=$WORLD_SIZE --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --use_env train.py
#python -m torch.distributed.launch --nproc_per_node=$GPU_NUM --nnodes=$WORLD_SIZE --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --use_env train_pipe.py
# juicesync src dst
# juicefs rmr your_dir
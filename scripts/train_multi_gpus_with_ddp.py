import os
import sys
# sys.path.append(sys.path[0]+'/../')

os.system(
    "CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 "
    "torchrun "
    "--nnodes=1 "
    "--nproc_per_node=6 "
    "train_inr_funsr_ddp.py"
)
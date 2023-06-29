import torch
import torchvision
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, CPUOffload
from torch.distributed.fsdp.wrap import wrap, size_based_auto_wrap_policy
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoTokenizer, GPT2TokenizerFast
from transformers import T5Tokenizer, T5ForConditionalGeneration
import functools
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from transformers.models.t5.modeling_t5 import T5Block

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
 checkpoint_wrapper,
 CheckpointImpl)

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    enable_wrap,
    wrap,
)
from functools import partial
from torch.utils.data import DataLoader
from pathlib import Path
from transformers.models.t5.modeling_t5 import T5Block
from typing import Type
import time
import tqdm
from datetime import datetime


def setup():
    # initialize the process group
    dist.init_process_group("nccl")

def cleanup():
    dist.destroy_process_group()


def fsdp_main(args):

    model = torchvision.models.resnet152()
    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    setup()
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    # sharding_strategy: ShardingStrategy = ShardingStrategy.SHARD_GRAD_OP  #for Zero2 and FULL_SHARD for Zero3
    # torch.cuda.set_device(local_rank)

    # model is on CPU before input to FSDP
    # cpu_offload = CPUOffload
    # cpu_offload.offload_params = True

    my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=100
    )
    cpu_offload = CPUOffload(offload_params=True)

    model = FSDP(
        model,
        # cpu_offload=cpu_offload,
        # auto_wrap_policy=my_auto_wrap_policy,
        # auto_wrap_policy=t5_auto_wrap_policy,
        # mixed_precision=mp_policy,
        #sharding_strategy=sharding_strategy,
        device_id=torch.cuda.current_device()
        )

    optimizer = optim.AdamW(model.parameters(), lr=0.001)


    for epoch in range(100):
        print(f"Epoch {epoch}")
        model.train()
        local_rank = int(os.environ['LOCAL_RANK'])
        optimizer.zero_grad()
        x = torch.rand(4, 3, 1024, 1024).to(device)
        output = model(x)
        loss = torch.sum(output)
        loss.backward()
        optimizer.step()
    dist.barrier()
    cleanup()


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch T5 FSDP Example')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    args = parser.parse_args()
    fsdp_main(args)


# torchrun --nnodes 1 --nproc_per_node 4  pytest.py
# 18226MiB
# 18650MiB
# 17778MiB

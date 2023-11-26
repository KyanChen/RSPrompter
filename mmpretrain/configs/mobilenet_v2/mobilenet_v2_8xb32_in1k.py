# Copyright (c) OpenMMLab. All rights reserved.
# This is a BETA new format config file, and the usage may change recently.
from mmengine.config import read_base

with read_base():
    from .._base_.datasets.imagenet_bs32_pil_resize import *
    from .._base_.default_runtime import *
    from .._base_.models.mobilenet_v2_1x import *
    from .._base_.schedules.imagenet_bs256_epochstep import *

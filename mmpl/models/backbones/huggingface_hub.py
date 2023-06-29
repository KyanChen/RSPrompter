import torch.nn as nn
from mmengine.model import BaseModule
from mmengine.model.weight_init import constant_init
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm
from mmpl.registry import MODELS
from .base_backbone import BaseBackbone


@MODELS.register_module()
class HuggingfaceModel(BaseBackbone):
    def __init__(self, ):
        pass

import torch.nn as nn
from mmengine.model import BaseModule
from mmengine.model.weight_init import constant_init
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm
from mmpl.registry import MODELS
from .base_backbone import BaseBackbone
from transformers import GPT2LMHeadModel, GPT2Config


@MODELS.register_module()
class HuggingfaceGPT(BaseBackbone):
    def __init__(self, model_name='gpt', from_pretrained=True):
        super(HuggingfaceGPT, self).__init__()
        self.model_name = model_name
        if from_pretrained:
            self.gpt_model = GPT2LMHeadModel.from_pretrained(model_name)
        else:
            self.gpt_model = GPT2LMHeadModel(config=GPT2Config.from_pretrained(model_name))

    def forward(self, *args, **kwargs):
        return self.gpt_model(*args, **kwargs)

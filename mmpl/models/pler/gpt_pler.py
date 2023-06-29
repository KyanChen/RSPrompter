from typing import Any

import torch
import torch.nn as nn
from mmpl.registry import MODELS
from ..builder import build_backbone, build_loss
from .base_pler import BasePLer
from mmpl.structures import ClsDataSample
from .base import BaseClassifier
import lightning.pytorch as pl
import torch.nn.functional as F


@MODELS.register_module()
class GPTPLer(BasePLer):
    def __init__(self,
                 backbone,
                 loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.backbone = build_backbone(backbone)
        self.loss = build_loss(loss)

    def training_step(self, batch, batch_idx):
        x, gt_label = batch['x'], batch['gt_label']
        outputs = self(input_ids=x, labels=gt_label)
        loss, logits = outputs['loss'], outputs['logits']
        return loss

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.backbone(*args, **kwargs)

    def validation_step(self, batch, batch_idx):
        pass

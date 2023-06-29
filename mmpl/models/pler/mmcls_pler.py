import os
from typing import Any

import mmengine
import torch
import torch.nn as nn
from einops import rearrange

from mmdet.models.utils import samplelist_boxtype2tensor
from mmdet.structures import SampleList
from mmdet.utils import InstanceList
from mmpl.registry import MODELS
from ..builder import build_backbone, build_loss, build_neck, build_head
from .base_pler import BasePLer
from mmpl.structures import ClsDataSample
from .base import BaseClassifier
import lightning.pytorch as pl
import torch.nn.functional as F


@MODELS.register_module()
class MMClsPLer(BasePLer):
    def __init__(self,
                 whole_model=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.whole_model = MODELS.build(whole_model)

    def setup(self, stage: str) -> None:
        super().setup(stage)

    def validation_step(self, batch, batch_idx):
        data = self.whole_model.data_preprocessor(batch, False)
        batch_data_samples = self.whole_model._run_forward(data, mode='predict')  # type: ignore

        pred_label = torch.cat([data_sample.pred_label for data_sample in batch_data_samples])
        gt_label = torch.cat([data_sample.gt_label for data_sample in batch_data_samples])
        self.val_evaluator.update(pred_label, gt_label)
        # self.val_evaluator.update(batch, batch_data_samples)

    def training_step(self, batch, batch_idx):
        data = self.whole_model.data_preprocessor(batch, True)
        losses = self.whole_model._run_forward(data, mode='loss')  # type: ignore
        parsed_losses, log_vars = self.parse_losses(losses)
        log_vars = {f'train_{k}': v for k, v in log_vars.items()}
        log_vars['loss'] = parsed_losses
        self.log_dict(log_vars, prog_bar=True)
        return log_vars

    def test_step(self, batch, batch_idx, *args: Any, **kwargs: Any):
        data = self.whole_model.data_preprocessor(batch, False)
        batch_data_samples = self.whole_model._run_forward(data, mode='predict')  # type: ignore
        self.test_evaluator.update(batch, batch_data_samples)







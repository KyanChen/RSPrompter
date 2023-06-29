import glob
import os
import time
from typing import Any
import cv2
import einops
import mmcv
import mmengine
import imageio
import numpy as np
import torch
from matplotlib import pyplot as plt
from mmpl.registry import HOOKS
from lightning.pytorch.callbacks import Callback


@HOOKS.register_module()
class SegVisualizer(Callback):
    def __init__(self, save_dir, *args, **kwargs):
        self.save_dir = save_dir
        mmengine.mkdir_or_exist(self.save_dir)

    def on_predict_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        data_samples: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        img = batch['inputs']
        pred = [data_sample.pred_sem_seg.data for data_sample in data_samples]
        label = [data_sample.gt_sem_seg.data for data_sample in data_samples]
        for i in range(len(img)):
            img_ = img[i].permute(1, 2, 0).cpu().numpy()
            pred_ = 255 * pred[i].permute(1, 2, 0).squeeze(-1).cpu().numpy()
            label_ = 255 * label[i].permute(1, 2, 0).squeeze(-1).cpu().numpy()
            pred_ = einops.repeat(pred_, 'h w -> h w c', c=3)
            label_ = einops.repeat(label_, 'h w -> h w c', c=3)
            cvss = np.hstack([img_, pred_, label_])
            cvss = cvss.astype(np.uint8)

            img_path = f"{self.save_dir}/{batch_idx*len(img)+i}_pred_gt.gif"
            imageio.imsave(img_path, cvss)


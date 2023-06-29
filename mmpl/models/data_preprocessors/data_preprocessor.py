import random
from numbers import Number
from typing import List, Optional, Sequence, Tuple, Union

import mmengine
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.dist import barrier, broadcast, get_dist_info
from mmengine.logging import MessageHub
from mmengine.model import BaseDataPreprocessor, ImgDataPreprocessor
from mmengine.structures import PixelData
from mmengine.utils import is_seq_of
from torch import Tensor

from mmdet.models.utils import unfold_wo_center
from mmdet.models.utils.misc import samplelist_boxtype2tensor
from mmpl.registry import MODELS
from mmdet.structures import DetDataSample
from mmdet.structures.mask import BitmapMasks
from mmdet.utils import ConfigType

try:
    import skimage
except ImportError:
    skimage = None


@MODELS.register_module()
class BatchFixedSizePadTokenMaskGPT(BaseDataPreprocessor):
    """Fixed size padding for batch images.

    Args:
        size (Tuple[int, int]): Fixed padding size. Expected padding
            shape (h, w). Defaults to None.
        img_pad_value (int): The padded pixel value for images.
            Defaults to 0.
        pad_mask (bool): Whether to pad instance masks. Defaults to False.
        mask_pad_value (int): The padded pixel value for instance masks.
            Defaults to 0.
        pad_seg (bool): Whether to pad semantic segmentation maps.
            Defaults to False.
        seg_pad_value (int): The padded pixel value for semantic
            segmentation maps. Defaults to 255.
    """

    def __init__(self,
                 pad_token: int,
                 p_token_keep: float = 1.,
                 nb_code: int = 512,
                 ) -> None:
        super().__init__()
        self.pad_token = pad_token
        self.p_token_keep = p_token_keep
        self.nb_code = nb_code

    def forward(
        self,
        batch
    ):
        # padding the input index to the same length

        longest = max([len(item) for item in batch['motion_token']])
        bs = len(batch['motion_token'])

        attention_mask = torch.zeros(bs, longest, dtype=torch.long, device=self.device)
        input_ids = torch.ones(bs, longest, dtype=torch.long, device=self.device) * self.pad_token
        for i, item in enumerate(batch['motion_token']):
            input_ids[i, :len(item)] = item
            attention_mask[i, :len(item)] = 1

        tgt_ids = input_ids

        if self.p_token_keep == -1:
            proba = np.random.rand(1)[0]
            mask = torch.bernoulli(proba * torch.ones(input_ids.shape,
                                                      device=input_ids.device))
        else:
            mask = torch.bernoulli(self.p_token_keep * torch.ones(input_ids.shape, device=input_ids.device))
        mask = mask.bool()
        r_indices = torch.randint_like(input_ids, self.nb_code)
        a_indices = mask * input_ids + mask.logical_not() * r_indices

        tgt_ids[tgt_ids == self.pad_token] = -100

        data = dict()
        data['inputs'] = dict(
            input_ids=a_indices,
            attention_mask=attention_mask,
            labels=tgt_ids,
        )
        data['data_samples'] = batch
        return data


@MODELS.register_module()
class NormalizationMotion(BaseDataPreprocessor):

    def __init__(
            self,
            mean_std_file: str,
    ) -> None:
        super().__init__()
        self.mean_std_info = mmengine.load(mean_std_file)

    def forward(
            self,
            batch
    ):
        for k, v in self.mean_std_info.items():
            for kk, vv in v.items():
                self.mean_std_info[k][kk] = vv.to(self.device, dtype=torch.float32)

        gt_motion = batch['motion']
        gt_motion = (gt_motion - self.mean_std_info['motion']['mean']) / self.mean_std_info['motion']['std']

        data = dict(
            inputs=gt_motion,
            data_samples=batch
        )
        return data

    def denormalize(self, x):
        return x * self.mean_std_info['motion']['std'] + self.mean_std_info['motion']['mean']
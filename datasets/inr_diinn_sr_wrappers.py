import copy
import functools
import os
import random
import math
from PIL import Image

import numpy as np
import torch
from einops import rearrange
from torch.utils.data import Dataset
from torchvision import transforms

from datasets import register
from utils import to_pixel_samples, to_coordinates

import torchvision.transforms.functional as TF
import random
from typing import Sequence


class MyRotateTransform:
    def __init__(self, angles: Sequence[int], p=0.5):
        self.angles = angles
        self.p = p

    def __call__(self, x):
        if torch.rand(1) < self.p:
            return x
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)


@register('inr_diinn_select_scale_sr_warp')
class INRSelectScaleSRWarp(Dataset):
    def __init__(self,
                 dataset, scales, patch_size=48,
                 augment=False,
                 val_mode=False, test_mode=False
                 ):
        super(INRSelectScaleSRWarp, self).__init__()
        self.dataset = dataset
        self.scales = scales
        self.patch_size = patch_size
        self.augment = augment
        self.test_mode = test_mode
        self.val_mode = val_mode

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # import pdb
        # pdb.set_trace()
        img_hr_ori, file_name = self.dataset[idx]
        class_name = os.path.basename(os.path.dirname(file_name))

        sample = {}
        for scale in self.scales:
            hr_size = self.patch_size * scale
            hr_size = int(hr_size)

            if self.test_mode or self.val_mode:
                hr_size = int(self.patch_size * max(self.scales))
                img_hr = transforms.CenterCrop(hr_size)(img_hr_ori)
            else:
                img_hr = transforms.RandomCrop(hr_size)(copy.deepcopy(img_hr_ori))
                if self.augment:
                    img_hr = transforms.RandomHorizontalFlip(p=0.5)(img_hr)
                    img_hr = transforms.RandomVerticalFlip(p=0.5)(img_hr)
                    img_hr = MyRotateTransform([90, 180, 270], p=0.5)(img_hr)

            img_lr = transforms.Resize(self.patch_size, TF.InterpolationMode.BICUBIC)(img_hr)
            sample[scale] = {'img': img_lr, 'gt': img_hr, 'class_name': class_name}

        return sample
import functools
import os.path
import random
import math

import torchvision.transforms
from PIL import Image
import numpy as np
import torch
from einops import rearrange
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from datasets import register
import torchvision.transforms
from utils import to_pixel_samples, to_coordinates



def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, Image.BICUBIC)(
            transforms.ToPILImage()(img)))


@register('cnn_fixed_scale_sr_warp')
class CNNFixedScaleSRWarp(Dataset):
    def __init__(self, dataset, scale_ratio, patch_size=48,
                 augment=False, val_mode=False, test_mode=False,
                 vis_continuous=False):
        self.dataset = dataset
        self.augment = augment
        self.scale_ratio = scale_ratio
        self.hr_size = int(patch_size * scale_ratio)
        self.test_mode = test_mode
        self.val_mode = val_mode
        self.patch_size = patch_size
        self.vis_continuous = vis_continuous

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_hr, file_name = self.dataset[idx]
        class_name = os.path.basename(os.path.dirname(file_name))
        file_name = os.path.basename(file_name).split('.')[0]

        if self.vis_continuous:
            img_lr = transforms.Resize(self.patch_size, InterpolationMode.BICUBIC)(
                transforms.CenterCrop(4*self.patch_size)(img_hr))

        # img_hr: 3xHxW
        if self.test_mode:
            img_hr = transforms.CenterCrop(self.hr_size)(img_hr)
        else:
            img_hr = transforms.RandomCrop(self.hr_size)(img_hr)

        if not self.vis_continuous:
            img_lr = transforms.Resize(self.patch_size, InterpolationMode.BICUBIC)(img_hr)

        if self.augment and not self.test_mode:
            if random.random() < 0.5:
                img_lr = img_lr.flip(-1)
                img_hr = img_hr.flip(-1)
            if random.random() < 0.5:
                img_lr = img_lr.flip(-2)
                img_hr = img_hr.flip(-2)

        return {
            'img': img_lr,
            'gt': img_hr,
            'class_name': class_name,
            'filename': file_name
        }

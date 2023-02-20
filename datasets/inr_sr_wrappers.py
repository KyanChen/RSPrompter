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

@register('inr_fixed_scale_sr_warp')
class INRFixedScaleSRWarp(Dataset):
    def __init__(self,
                 dataset, scale_ratio, patch_size=48,
                 augment=False, sample_q=None,
                 val_mode=False, test_mode=False,
                 encode_scale_ratio=False,
                 return_cell=False,  # for liff
                 ):
        super(INRFixedScaleSRWarp, self).__init__()
        self.dataset = dataset
        self.scale_ratio = scale_ratio
        self.patch_size = patch_size
        self.hr_size = int(patch_size * scale_ratio)
        self.augment = augment
        self.sample_q = sample_q
        self.test_mode = test_mode
        self.val_mode = val_mode
        self.encode_scale_ratio = encode_scale_ratio
        self.return_cell = return_cell

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # import pdb
        # pdb.set_trace()
        img_hr, file_name = self.dataset[idx]
        class_name = os.path.basename(os.path.dirname(file_name))
        file_name = os.path.basename(file_name).split('.')[0]
        # img_hr: 3xHxW
        h, w = img_hr.shape[-2:]
        # if h < 256 or w < 256:
        #     img_hr = transforms.Resize(256, Image.BICUBIC)(img_hr)

        if self.test_mode or self.val_mode:
            img_hr = transforms.CenterCrop(self.hr_size)(img_hr)
        else:
            img_hr = transforms.RandomCrop(self.hr_size)(img_hr)
            if self.augment:
                img_hr = transforms.RandomHorizontalFlip(p=0.5)(img_hr)
                img_hr = transforms.RandomVerticalFlip(p=0.5)(img_hr)
                img_hr = MyRotateTransform([90, 180, 270], p=0.5)(img_hr)

        img_lr = transforms.Resize(self.patch_size, Image.BICUBIC)(img_hr)

        hr_coord = to_coordinates(size=img_hr.shape[-2:], return_map=False)
        hr_rgb = rearrange(img_hr, 'C H W -> (H W) C')

        if self.sample_q is not None and not self.test_mode:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]
        return_dict = {
            'inp': img_lr,
            'coord': hr_coord,
            'gt': hr_rgb,
            'class_name': class_name,
            'filename': file_name
        }

        if self.encode_scale_ratio:
            scale_ratio = torch.ones_like(hr_coord) * self.patch_size / self.hr_size
            return_dict['scale_ratio'] = scale_ratio

        if self.return_cell:
            cell = torch.ones_like(hr_coord)
            cell[:, 0] *= 2 / img_hr.shape[-2]
            cell[:, 1] *= 2 / img_hr.shape[-1]
            return_dict['cell'] = cell

        return return_dict


@register('inr_range_scale_sr_warp')
class INRRangeScaleSRWarp(Dataset):
    def __init__(self,
                 dataset, max_scale_ratio, patch_size=48,
                 augment=False, sample_q=None,
                 val_mode=False, test_mode=False,
                 encode_scale_ratio=False,
                 return_cell=False,  # for liff
                 ):
        super(INRRangeScaleSRWarp, self).__init__()
        self.dataset = dataset
        self.max_scale_ratio = max_scale_ratio
        self.patch_size = patch_size
        assert max_scale_ratio <= 8
        self.augment = augment
        self.sample_q = sample_q
        self.test_mode = test_mode
        self.val_mode = val_mode
        self.encode_scale_ratio = encode_scale_ratio
        self.return_cell = return_cell

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_hr, file_name = self.dataset[idx]
        class_name = os.path.basename(os.path.dirname(file_name))
        h, w = img_hr.shape[-2:]
        # if h < 256 or w < 256:
        #     img_hr = transforms.Resize(256, Image.BICUBIC)(img_hr)

        hr_size = self.patch_size + self.patch_size * torch.rand([]) * (self.max_scale_ratio - 1)
        hr_size = int(hr_size)

        if self.test_mode or self.val_mode:
            hr_size = int(self.patch_size * self.max_scale_ratio)
            img_hr = transforms.CenterCrop(hr_size)(img_hr)
        else:
            img_hr = transforms.RandomCrop(hr_size)(img_hr)
            if self.augment:
                img_hr = transforms.RandomHorizontalFlip(p=0.5)(img_hr)
                img_hr = transforms.RandomVerticalFlip(p=0.5)(img_hr)
                img_hr = MyRotateTransform([90, 180, 270], p=0.5)(img_hr)

        img_lr = transforms.Resize(self.patch_size, Image.BICUBIC)(img_hr)

        hr_coord = to_coordinates(size=img_hr.shape[-2:], return_map=False)
        hr_rgb = rearrange(img_hr, 'C H W -> (H W) C')

        if self.sample_q is not None and not self.test_mode:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]
        return_dict = {
            'inp': img_lr,
            'coord': hr_coord,
            'gt': hr_rgb,
            'class_name': class_name
        }
        if self.encode_scale_ratio:
            scale_ratio = torch.ones_like(hr_coord) * self.patch_size / hr_size
            return_dict['scale_ratio'] = scale_ratio

        if self.return_cell:
            cell = torch.ones_like(hr_coord)
            cell[:, 0] *= 2 / img_hr.shape[-2]
            cell[:, 1] *= 2 / img_hr.shape[-1]
            return_dict['cell'] = cell

        return return_dict

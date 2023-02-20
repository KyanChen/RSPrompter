import functools
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


def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, Image.BICUBIC)(
            transforms.ToPILImage()(img)))


@register('rs_sr_warp')
class RSSRWarp(Dataset):
    def __init__(self, dataset, size_min=None, size_max=None,
                 augment=False, gt_resize=None, sample_q=None, val_mode=False):
        self.dataset = dataset
        self.size_min = size_min
        if size_max is None:
            size_max = size_min
        self.size_max = size_max
        self.augment = augment
        self.gt_resize = gt_resize
        self.sample_q = sample_q
        self.val_mode = val_mode

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr = self.dataset[idx]
        # p = idx / (len(self.dataset) - 1)
        if not self.val_mode:
            p = random.random()
            w_hr = round(self.size_min + (self.size_max - self.size_min) * p)
            img_hr = resize_fn(img_hr, w_hr)
        else:
            img_hr = resize_fn(img_hr, self.size_max)


        if self.augment and not self.val_mode:
            if random.random() < 0.5:
                img_lr = img_lr.flip(-1)
                img_hr = img_hr.flip(-1)
            if random.random() < 0.5:
                img_lr = img_lr.flip(-2)
                img_hr = img_hr.flip(-2)

        if self.gt_resize is not None:
            img_hr = resize_fn(img_hr, self.gt_resize)

        hr_coord = to_coordinates(size=img_hr.shape[-2:], return_map=False)
        hr_rgb = rearrange(img_hr, 'C H W -> (H W) C')

        if self.sample_q is not None:
            sample_lst = np.random.choice(len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        # cell = torch.ones_like(hr_coord)
        # cell[:, 0] *= 2 / img_hr.shape[-2]
        # cell[:, 1] *= 2 / img_hr.shape[-1]

        return {
            'inp': img_lr,
            'coord': hr_coord,
            'gt': hr_rgb
        }

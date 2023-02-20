import os
import time
import shutil
import math

import cv2
import torch
import numpy as np
from einops import rearrange
from torch.optim import SGD, Adam, AdamW
from tensorboardX import SummaryWriter
import torch.nn.functional as F


def warm_up_cosine_lr_scheduler(optimizer, epochs=100, warm_up_epochs=5, eta_min=1e-9):
    """
    Description:
        - Warm up cosin learning rate scheduler, first epoch lr is too small

    Arguments:
        - optimizer: input optimizer for the training
        - epochs: int, total epochs for your training, default is 100. NOTE: you should pass correct epochs for your training
        - warm_up_epochs: int, default is 5, which mean the lr will be warm up for 5 epochs. if warm_up_epochs=0, means no need
          to warn up, will be as cosine lr scheduler
        - eta_min: float, setup ConsinAnnealingLR eta_min while warm_up_epochs = 0

    Returns:
        - scheduler
    """

    if warm_up_epochs <= 0:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)

    else:
        warm_up_with_cosine_lr = lambda epoch: eta_min + (epoch / warm_up_epochs) \
            if epoch <= warm_up_epochs else \
            0.5 * (np.cos((epoch - warm_up_epochs) / (epochs - warm_up_epochs) * np.pi) + 1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)

    return scheduler


class Averager():

    def __init__(self, class_names=['all']):
        if 'all' not in class_names:
            class_names.append('all')
        self.values = {k: [] for k in class_names}

    def add(self, ks, vs):
        if torch.is_tensor(vs):
            vs = vs.cpu().tolist()
        for k, v in zip(ks, vs):
            self.values[k].append(v)
            self.values['all'].append(v)

    def item(self):
        return_dict = {}
        for k, v in self.values.items():
            if len(v):
                return_dict[k] = sum(v) / len(v)
            else:
                return_dict[k] = 0
        return return_dict

class AveragerList():

    def __init__(self):
        self.values = []

    def add(self, vs):
        if torch.is_tensor(vs):
            vs = vs.cpu().tolist()
        if isinstance(vs, list):
            self.values += vs
        else:
            self.values += [vs]

    def item(self):
        return sum(self.values) / len(self.values)


class Timer():

    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v


def time_text(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    elif t >= 60:
        return '{:.1f}m'.format(t / 60)
    else:
        return '{:.1f}s'.format(t)


_log_path = None


def set_log_path(path):
    global _log_path
    _log_path = path


def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)


def ensure_path(path, remove=True):
    basename = os.path.basename(path.rstrip('/'))
    if os.path.exists(path):
        print('{} exists!'.format(path))
        # if remove and (basename.startswith('_')
        #         or input('{} exists, remove? (y/[n]): '.format(path)) == 'y'):
        #     shutil.rmtree(path)
        #     os.makedirs(path)
    else:
        os.makedirs(path)


def set_save_path(save_path, remove=True):
    ensure_path(save_path, remove=remove)
    set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))
    return log, writer


def compute_num_params(model, text=False):
    tot = int(sum([np.prod(p.shape) for p in model.parameters()]))
    if text:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot


def make_optimizer(param_list, optimizer_spec, load_sd=False):
    Optimizer = {
        'sgd': SGD,
        'adam': Adam,
        'adamw': AdamW,
    }[optimizer_spec['name']]
    default_args = {
        'sgd': {},
        'adam':
            {
            'betas': (0.9, 0.999),
            'eps': 1e-08,
            'weight_decay': 0,
            'amsgrad': False
            },
        'adamw': {},
    }[optimizer_spec['name']]
    default_args.update(optimizer_spec['args'])
    optimizer = Optimizer(param_list, **default_args)
    if load_sd:
        optimizer.load_state_dict(optimizer_spec['sd'])
    return optimizer


def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


def to_coordinates(size=(56, 56), return_map=True):
    """Converts an image to a set of coordinates and features.

    Args:
        img (torch.Tensor): Shape (channels, height, width).
    """
    # H, W
    # Coordinates are indices of all non zero locations of a tensor of ones of
    # same shape as spatial dimensions of image
    coordinates = torch.ones(size).nonzero(as_tuple=False).float()
    # Normalize coordinates to lie in [-.5, .5]
    coordinates[..., 0] = coordinates[..., 0] / (size[0] - 1) - 0.5
    coordinates[..., 1] = coordinates[..., 1] / (size[1] - 1) - 0.5
    # Convert to range [-1, 1]
    coordinates *= 2
    if return_map:
        coordinates = rearrange(coordinates, '(H W) C -> H W C', H=size[0])
    # [y, x]
    return coordinates


def to_pixel_samples(img):
    """ Convert the image to coord-RGB pairs.
        img: Tensor, (3, H, W)
    """
    coord = make_coord(img.shape[-2:])
    rgb = img.view(3, -1).permute(1, 0)
    return coord, rgb


def get_clamped_psnr(img, img_recon, rgb_range=1, crop_border=None):
    # Values may lie outside [0, 1], so clamp input
    img_recon = torch.clamp(img_recon, 0., 1.)
    # Pixel values lie in {0, ..., 255}, so round float tensor
    img_recon = torch.round(img_recon * 255) / 255.
    diff = img - img_recon
    if crop_border is not None:
        assert len(diff.size()) == 4
        valid = diff[..., crop_border:-crop_border, crop_border:-crop_border]
    else:
        valid = diff

    psnr_list = []
    for i in range(len(img)):
        psnr = 20. * np.log10(1.) - 10. * valid[i].detach().pow(2).mean().log10().to('cpu').item()
        psnr_list.append(psnr)
    return psnr_list


def _ssim_pth(img, img2):
    """Calculate SSIM (structural similarity) (PyTorch version).
    It is called by func:`calculate_ssim_pt`.
    Args:
        img (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        img2 (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
    Returns:
        float: SSIM result.
    """
    c1 = (0.01 * 255)**2
    c2 = (0.03 * 255)**2

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    window = torch.from_numpy(window).view(1, 1, 11, 11).expand(img.size(1), 1, 11, 11).to(img.dtype).to(img.device)

    mu1 = F.conv2d(img, window, stride=1, padding=0, groups=img.shape[1])  # valid mode
    mu2 = F.conv2d(img2, window, stride=1, padding=0, groups=img2.shape[1])  # valid mode
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img * img, window, stride=1, padding=0, groups=img.shape[1]) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, stride=1, padding=0, groups=img.shape[1]) - mu2_sq
    sigma12 = F.conv2d(img * img2, window, stride=1, padding=0, groups=img.shape[1]) - mu1_mu2

    cs_map = (2 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
    ssim_map = ((2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)) * cs_map
    return ssim_map.mean([1, 2, 3])


def calculate_ssim_pt(img, img2, crop_border, test_y_channel=False, **kwargs):
    """Calculate SSIM (structural similarity) (PyTorch version).
    ``Paper: Image quality assessment: From error visibility to structural similarity``
    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.
    For three-channel images, SSIM is calculated for each channel and then
    averaged.
    Args:
        img (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        img2 (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.
    Returns:
        float: SSIM result.
    """

    assert img.shape == img2.shape, f'Image shapes are different: {img.shape}, {img2.shape}.'

    if crop_border != 0:
        img = img[:, :, crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]

    if test_y_channel:
        img = rgb2ycbcr_pt(img, y_only=True)
        img2 = rgb2ycbcr_pt(img2, y_only=True)

    img = img.to(torch.float64)
    img2 = img2.to(torch.float64)

    ssim = _ssim_pth(img * 255., img2 * 255.)
    return ssim


def calculate_psnr_pt(img, img2, crop_border, test_y_channel=False, **kwargs):
    """Calculate PSNR (Peak Signal-to-Noise Ratio) (PyTorch version).
    Reference: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    Args:
        img (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        img2 (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.
    Returns:
        float: PSNR result.
    """

    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')

    if crop_border != 0:
        img = img[:, :, crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]

    if test_y_channel:
        img = rgb2ycbcr_pt(img, y_only=True)
        img2 = rgb2ycbcr_pt(img2, y_only=True)

    img = img.to(torch.float64)
    img2 = img2.to(torch.float64)

    mse = torch.mean((img - img2)**2, dim=[1, 2, 3])
    return 10. * torch.log10(1. / (mse + 1e-8))

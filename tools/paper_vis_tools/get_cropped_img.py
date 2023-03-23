import glob
import os
import cv2
import matplotlib.pyplot as plt
import torch

import utils


img_name = 'agricultural48'
folder = '/Users/kyanchen/Downloads/funsr'
save_folder = f'/Users/kyanchen/Downloads/{img_name}'

os.makedirs(save_folder, exist_ok=True)
crop = [55, 28, 118, 85]

# get LR and HR
img_file = 'vis_FunSR-RDN_UC_4x_testset'
lr_file = folder+f'/{img_file}/{img_name}_Ori.png'
hr_file = folder+f'/{img_file}/{img_name}_GT.png'
lr = cv2.imread(lr_file)
hr = cv2.imread(hr_file)
hr_copy = hr.copy()
lr = cv2.resize(lr, (hr.shape[1], hr.shape[0]))
lr = lr[crop[1]:crop[3], crop[0]:crop[2]]
cv2.imwrite(save_folder+f'/{os.path.basename(os.path.dirname(lr_file))}_{os.path.basename(lr_file)}', lr)

cv2.rectangle(hr, crop[:2], crop[2:], (0, 0, 255), 1)
cv2.imwrite(save_folder+f'/{os.path.basename(os.path.dirname(hr_file))}_{os.path.basename(hr_file)}', hr)

hr_copy = hr_copy[crop[1]:crop[3], crop[0]:crop[2]]


all_4x_imgfiles = glob.glob(folder+f'/*/{img_name}_4.0X*')
metric_fn = [utils.calculate_psnr_pt, utils.calculate_ssim_pt]

for img_file in all_4x_imgfiles:
    img = cv2.imread(img_file)
    img = img[crop[1]:crop[3], crop[0]:crop[2]]

    pred = torch.from_numpy(img).unsqueeze(0).permute((0, 3, 1, 2)) / 255.
    gt = torch.from_numpy(hr_copy).unsqueeze(0).permute((0, 3, 1, 2)) / 255.
    psnr = metric_fn[0](
                pred,
                gt,
                crop_border=0
            )[0]
    ssim = metric_fn[1](
        pred,
        gt,
        crop_border=0
    )[0]

    cv2.imwrite(save_folder + f'/{os.path.basename(os.path.dirname(img_file))}_{os.path.basename(img_file).split("_")[0]}_{psnr:.2f}_{ssim:.4f}.png', img)

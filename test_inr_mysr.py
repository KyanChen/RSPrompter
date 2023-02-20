import argparse
import json
import os

import math
from functools import partial

import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import models
import utils

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def batched_predict(model, inp, coord, bsize):
    with torch.no_grad():
        pred = model(inp, coord)
    return pred


def eval_psnr(loader, class_names, model, data_norm=None, eval_type=None, eval_bsize=None, verbose=False, crop_border=4):
    crop_border = int(crop_border) if crop_border else crop_border
    print('crop border: ', crop_border)
    model.eval()

    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).to(device)
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).to(device)
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).to(device)
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).to(device)

    if eval_type is None:
        metric_fn = [utils.calculate_psnr_pt, utils.calculate_ssim_pt]
    elif eval_type == 'psnr+ssim':
        metric_fn = [utils.calculate_psnr_pt, utils.calculate_ssim_pt]
    elif eval_type.startswith('div2k'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='div2k', scale=scale)
    elif eval_type.startswith('benchmark'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='benchmark', scale=scale)
    else:
        raise NotImplementedError

    val_res_psnr = utils.Averager(class_names)
    val_res_ssim = utils.Averager(class_names)

    pbar = tqdm(loader, leave=False, desc='val')
    for batch in pbar:
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(device)

        inp = (batch['inp'] - inp_sub) / inp_div
        # import pdb
        # pdb.set_trace()
        if eval_bsize is None:
            with torch.no_grad():
                scale_ratios = batch.get('scale_ratio', None)
                if scale_ratios is None:
                    pred = model(inp, batch['coord'])[-1]
                else:
                    # scale_ratios = (scale_ratios - gt_sub) / gt_div
                    pred = model(inp, batch['coord'], scale_ratios)[-1]
        else:
            pred = batched_predict(model, inp, batch['coord'], eval_bsize)
        pred = pred * gt_div + gt_sub

        if eval_type is not None: # reshape for shaving-eval
            ih, iw = batch['inp'].shape[-2:]
            s = math.sqrt(batch['coord'].shape[1] / (ih * iw))
            if s > 1:
                shape = [batch['inp'].shape[0], round(ih * s), round(iw * s), 3]
            else:
                shape = [batch['inp'].shape[0], 32, batch['coord'].shape[1]//32, 3]

            pred = pred.view(*shape) \
                .permute(0, 3, 1, 2).contiguous()
            batch['gt'] = batch['gt'].view(*shape) \
                .permute(0, 3, 1, 2).contiguous()

        # if crop_border is not None:
        #     h = math.sqrt(pred.shape[1])
        #     shape = [inp.shape[0], round(h), round(h), 3]
        #     pred = pred.view(*shape).permute(0, 3, 1, 2).contiguous()
        #     batch['gt'] = batch['gt'].view(*shape).permute(0, 3, 1, 2).contiguous()
        # else:
        #     pred = pred.permute(0, 2, 1).contiguous()  # B 3 N
        #     batch['gt'] = batch['gt'].permute(0, 2, 1).contiguous()

        res_psnr = metric_fn[0](
            pred,
            batch['gt'],
            crop_border=crop_border
        )
        res_ssim = metric_fn[1](
            pred,
            batch['gt'],
            crop_border=crop_border
        )

        val_res_psnr.add(batch['class_name'], res_psnr)
        val_res_ssim.add(batch['class_name'], res_ssim)

        if verbose:
            pbar.set_description(
                'val psnr: {:.4f} ssim: {:.4f}'.format(val_res_psnr.item()['all'], val_res_ssim.item()['all']))

    return val_res_psnr.item(), val_res_ssim.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/test_UC_INR_mysr.yaml')
    parser.add_argument('--model', default='checkpoints/EXP20220610_5/epoch-best.pth')
    # parser.add_argument('--model', default='checkpoints/EXP20220610_5/epoch-last.pth')
    parser.add_argument('--scale_ratio', default=4, type=float)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config['test_dataset']['wrapper']['args']['scale_ratio'] = args.scale_ratio

    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(dataset, batch_size=spec['batch_size'], num_workers=0, pin_memory=True, shuffle=False, drop_last=False)

    model_spec = torch.load(args.model)['model']
    print(model_spec['args'])
    model = models.make(model_spec, load_sd=True).to(device)

    file_names = json.load(open(config['test_dataset']['dataset']['args']['split_file']))['test']
    class_names = list(set([os.path.basename(os.path.dirname(x)) for x in file_names]))

    crop_border = config['test_dataset']['wrapper']['args']['scale_ratio']+5
    dataset_name = os.path.basename(config['test_dataset']['dataset']['args']['split_file']).split('_')[0]
    max_scale = {'UC': 5, 'AID': 12}
    if args.scale_ratio > max_scale[dataset_name]:
        crop_border = int((args.scale_ratio - max_scale[dataset_name]) / 2 * 48)

    res = eval_psnr(
        loader, class_names, model,
        data_norm=config.get('data_norm'),
        eval_type=config.get('eval_type'),
        eval_bsize=config.get('eval_bsize'),
        crop_border=crop_border,
        verbose=True)
    # print('psnr')
    # for k, v in res[0].items():
    #     print(f'{k}: {v:0.2f}')
    # print('ssim')
    # for k, v in res[1].items():
    #     print(f'{k}: {v:0.4f}')
    print(f'psnr: {res[0]["all"]:0.2f}')
    print(f'ssim: {res[1]["all"]:0.4f}')
import argparse
import json
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR

import datasets
import models
import utils


def make_data_loader(spec, tag='', local_rank=0):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    if local_rank == 0:
        print('{} dataset: size={}'.format(tag, len(dataset)))
        for k, v in dataset[0].items():
            if torch.is_tensor(v):
                print('  {}: shape={}'.format(k, v.shape))
            elif isinstance(v, str):
                pass
            elif isinstance(v, dict):
                for k0, v0 in v.items():
                    if hasattr(v0, 'shape'):
                        print('  {}: shape={}'.format(k0, v0.shape))
            else:
                raise NotImplementedError
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=(tag == 'train'))
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=spec['batch_size'],
                                         num_workers=spec['num_workers'],
                                         pin_memory=True,
                                         sampler=sampler)
    return loader


def make_data_loaders(config, local_rank):
    train_loader = make_data_loader(config.get('train_dataset'), tag='train', local_rank=local_rank)
    val_loader = make_data_loader(config.get('val_dataset'), tag='val', local_rank=local_rank)
    return train_loader, val_loader


def prepare_training(config, local_rank):
    if config.get('resume') is not None:
        sv_file = torch.load(config['resume'])
        model = models.make(sv_file['model'], load_sd=True).cuda()
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        optimizer = utils.make_optimizer(
            model.parameters(), sv_file['optimizer'], load_sd=True)
        epoch_start = sv_file['epoch'] + 1
        if config.get('multi_step_lr') is None:
            lr_scheduler = None
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])
        for _ in range(epoch_start - 1):
            lr_scheduler.step()
    else:
        model = models.make(config['model']).cuda(local_rank)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = 1
        lr_scheduler = config.get('lr_scheduler')
        lr_scheduler_name = lr_scheduler.pop('name')
        if 'MultiStepLR' == lr_scheduler_name:
            lr_scheduler = MultiStepLR(optimizer, **lr_scheduler)
        elif 'CosineAnnealingLR' == lr_scheduler_name:
            lr_scheduler = CosineAnnealingLR(optimizer, **lr_scheduler)
        elif 'CosineAnnealingWarmUpLR' == lr_scheduler_name:
            lr_scheduler = utils.warm_up_cosine_lr_scheduler(optimizer, **lr_scheduler)
    if local_rank == 0:
        print('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    return model, optimizer, epoch_start, lr_scheduler

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def return_avg(self):
        return self.avg

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def train(train_loader, model, optimizer, local_rank):
    model = model.train()
    loss_fn = nn.L1Loss().cuda(local_rank)
    train_losses = AverageMeter('Loss', ':.4e')

    data_norm = config['data_norm']
    t = data_norm['img']
    img_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda(local_rank)
    img_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda(local_rank)
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda(local_rank)
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda(local_rank)

    if local_rank == 0:
        pbar = tqdm(total=len(train_loader), desc='train', leave=False)

    for i, batch in enumerate(train_loader):
        if local_rank == 0:
            pbar.update(1)
        keys = list(batch.keys())
        batch = batch[keys[torch.randint(0, len(keys), [])]]
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.cuda(local_rank, non_blocking=True)
        img = (batch['img'] - img_sub) / img_div
        gt = (batch['gt'] - gt_sub) / gt_div
        pred = model(img, gt.shape[-2:])
        if isinstance(pred, tuple):
            loss = 0.2 * loss_fn(pred[0], gt) + loss_fn(pred[1], gt)
        elif isinstance(pred, list):
            losses = [loss_fn(x, gt) for x in pred]
            losses = [x * (idx + 1) for idx, x in enumerate(losses)]
            loss = sum(losses) / ((1 + len(losses)) * len(losses) / 2)
        else:
            loss = loss_fn(pred, gt)

        torch.distributed.barrier()
        reduced_loss = reduce_mean(loss, dist.get_world_size())
        train_losses.update(reduced_loss.item(), img.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if local_rank == 0:
        pbar.close()
    return train_losses.avg


def eval_psnr(loader, class_names, model, local_rank, data_norm=None, eval_type=None, eval_bsize=None, verbose=False, crop_border=4):
    crop_border = int(crop_border) if crop_border else crop_border
    if local_rank == 0:
        print('crop border: ', crop_border)
    model = model.eval()

    if data_norm is None:
        data_norm = {
            'img': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }
    t = data_norm['img']
    img_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda(local_rank)
    img_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda(local_rank)
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda(local_rank)
    gt_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda(local_rank)

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

    val_res_psnr = AverageMeter('psnr', ':.4f')
    val_res_ssim = AverageMeter('ssim', ':.4f')

    if local_rank == 0:
        pbar = tqdm(total=len(loader), desc='val', leave=False)
    for batch in loader:
        if local_rank == 0:
            pbar.update(1)
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.cuda(local_rank, non_blocking=True)

        img = (batch['img'] - img_sub) / img_div
        with torch.no_grad():
            pred = model(img, batch['gt'].shape[-2:])
        if isinstance(pred, list):
            pred = pred[-1]
        pred = pred * gt_div + gt_sub

        res_psnr = metric_fn[0](
            pred,
            batch['gt'],
            crop_border=crop_border
        ).mean()
        res_ssim = metric_fn[1](
            pred,
            batch['gt'],
            crop_border=crop_border
        ).mean()

        torch.distributed.barrier()
        reduced_val_res_psnr = reduce_mean(res_psnr, dist.get_world_size())
        reduced_val_res_ssim = reduce_mean(res_ssim, dist.get_world_size())

        val_res_psnr.update(reduced_val_res_psnr.item(), img.size(0))
        val_res_ssim.update(reduced_val_res_ssim.item(), img.size(0))

        if verbose and local_rank == 0:
            pbar.set_description(
                'val psnr: {:.4f} ssim: {:.4f}'.format(val_res_psnr.avg, val_res_ssim.avg))
    if local_rank == 0:
        pbar.close()
    return val_res_psnr.avg, val_res_ssim.avg


def main(config, save_path):
    # torch.backends.cudnn.benchmark = True
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    print(f'rank: {rank} local_rank: {local_rank} world_size: {world_size}')
    # print(f'local_rank: {torch.distributed.local_rank()}')
    if local_rank == 0:
        log, writer = utils.set_save_path(save_path)
        with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
            yaml.dump(config, f, sort_keys=False)

    train_loader, val_loader = make_data_loaders(config, local_rank)
    if config.get('data_norm') is None:
        config['data_norm'] = {
            'img': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }

    model, optimizer, epoch_start, lr_scheduler = prepare_training(config, local_rank)

    epoch_max = config['epoch_max']
    epoch_val_interval = config.get('epoch_val_interval')
    epoch_save_interval = config.get('epoch_save_interval')
    max_val_v = -1e18

    timer = utils.Timer()

    for epoch in range(epoch_start, epoch_max + 1):
        t_epoch_start = timer.t()
        train_loader.sampler.set_epoch(epoch)

        train_loss = train(train_loader, model, optimizer, local_rank)
        if lr_scheduler is not None:
            lr_scheduler.step()

        if rank == 0:
            log_info = ['epoch {}/{}'.format(epoch, epoch_max)]
            log_info.append('train: loss={:.4f}'.format(train_loss))
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalars('loss', {'train': train_loss}, epoch)

        model_ = model.module
        model_spec = config['model']
        model_spec['sd'] = model_.state_dict()
        optimizer_spec = config['optimizer']
        optimizer_spec['sd'] = optimizer.state_dict()
        sv_file = {
            'model': model_spec,
            'optimizer': optimizer_spec,
            'epoch': epoch
        }
        if rank == 0:
            torch.save(sv_file, os.path.join(save_path, 'epoch-last.pth'))

        if (epoch_save_interval is not None) and (epoch % epoch_save_interval == 0):
            if rank == 0:
                torch.save(sv_file, os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))

        if (epoch_val_interval is not None) and (epoch % epoch_val_interval == 0):
            file_names = json.load(open(config['val_dataset']['dataset']['args']['split_file']))['test']
            class_names = list(set([os.path.basename(os.path.dirname(x)) for x in file_names]))

            val_res_psnr, val_res_ssim = eval_psnr(val_loader, class_names, model_, local_rank,
                                                   data_norm=config['data_norm'],
                                                   eval_type=config.get('eval_type'),
                                                   eval_bsize=config.get('eval_bsize'),
                                                   crop_border=4)
            if rank == 0:
                log_info.append('val: psnr={:.4f}'.format(val_res_psnr))
                writer.add_scalars('psnr', {'val': val_res_psnr}, epoch)
            if val_res_psnr > max_val_v:
                max_val_v = val_res_psnr
                if rank == 0:
                    torch.save(sv_file, os.path.join(save_path, 'epoch-best.pth'))

        t = timer.t()
        if rank == 0:
            prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
            t_epoch = utils.time_text(t - t_epoch_start)
            t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
            log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))
            log(', '.join(log_info))
            writer.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/train_1x-5x_INR_funsr.yaml')
    parser.add_argument('--name', default='EXP20221216_11')
    parser.add_argument('--tag', default=None)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('./checkpoints', save_name)
    main(config, save_path)

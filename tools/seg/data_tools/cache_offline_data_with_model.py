import argparse
import os.path as osp
import sys
sys.path.insert(0, sys.path[0]+'/../..')
from mmpl.models import build_pler
import mmengine
import torch

from torch.nn import functional as F
from mmengine.config import Config, DictAction
from mmengine.utils import ProgressBar

from mmpl.datasets.builder import build_dataset
from mmpl.utils import register_all_modules


def parse_args():
    parser = argparse.ArgumentParser(description='Browse a dataset')
    parser.add_argument('--config', default='configs/seg/seg_just_sam_backbone_config.py', help='train config file path')
    parser.add_argument(
        '--output-dir',
        '-o',
        default='cache_data/sam_data',
        type=str,
        help='If there is no display interface, you can save it.')
    parser.add_argument(
        '--phase',
        '-p',
        default=['train', 'val'],
        help='phase of dataset to visualize, accept "train" "test" and "val".'
        ' Defaults to "train".')

    parser.add_argument(
        '--show-number',
        '-n',
        type=int,
        default=sys.maxsize,
        help='number of images selected to visualize, must bigger than 0. if '
        'the number is bigger than length of dataset, show all the images in '
        'dataset; default "sys.maxsize", show all images in dataset')
    parser.add_argument(
        '--show-interval',
        '-i',
        type=float,
        default=2,
        help='the interval of show (s)')
    parser.add_argument(
        '--mode',
        '-m',
        default='transformed',
        type=str,
        choices=['original', 'transformed', 'concat', 'pipeline'],
        help='display mode; display original pictures or transformed pictures'
        ' or comparison pictures. "original" means show images load from disk'
        '; "transformed" means to show images after transformed; "concat" '
        'means show images stitched by "original" and "output" images. '
        '"pipeline" means show all the intermediate images. '
        'Defaults to "transformed".')
    parser.add_argument(
        '--rescale-factor',
        '-r',
        type=float,
        help='image rescale factor, which is useful if the output is too '
        'large or too small.')
    parser.add_argument(
        '--channel-order',
        '-c',
        default='BGR',
        choices=['BGR', 'RGB'],
        help='The channel order of the showing images, could be "BGR" '
        'or "RGB", Defaults to "BGR".')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def init_model(cfg, device='cuda:0'):
    model = build_pler(cfg.model_cfg).to(device)
    model.eval()
    return model


def model_forward_save(all_results_list, model, output_dir, phase, device='cuda:0', local_rank_id=0,):
    pbar = ProgressBar(len(all_results_list))
    for i, results in enumerate(all_results_list):
        pbar.update()
        image = results['inputs'].unsqueeze(0).clone().detach().float().to(device)
        img = F.interpolate(image, size=(model.sam.img_size, model.sam.img_size), mode='bicubic', align_corners=False)
        img = img[:, [2, 1, 0], :, :]  # BGR2RGB
        img = (img - model.sam.pixel_mean) / model.sam.pixel_std
        with torch.no_grad():
            image_embeddings, inner_states = model.sam(img)  # Bx256x64x64
        img_path = results['data_samples'].img_path
        if isinstance(inner_states, list):
            inner_states = [inner_state.cpu() for inner_state in inner_states]
        # import ipdb; ipdb.set_trace()
        inner_states = [inner_state for idx, inner_state in enumerate(inner_states) if idx >= 32-4]
        cache_data = {'image_embeddings': image_embeddings.cpu(), 'inner_states': inner_states}
        torch.save(cache_data, f"{output_dir}/{phase}_{osp.splitext(osp.basename(img_path))[0]}.pt")
        # mmengine.dump(cache_data, f"{output_dir}/{phase}_{osp.splitext(osp.basename(img_path))[0]}.pkl")

def main():
    torch.multiprocessing.set_start_method('spawn', force=True)
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    mmengine.mkdir_or_exist(args.output_dir)
    # register all modules in mmcls into the registries
    register_all_modules()
    if isinstance(args.phase, str):
        phases = [args.phase]
    else:
        phases = args.phase
    is_cuda = torch.cuda.is_available()
    # model = init_model(cfg, device=device)

    cache_datasets = []
    for phase in phases:
        dataset_cfg = cfg.get('datamodule_cfg').get(phase + '_loader', None)
        if dataset_cfg is None:
            continue
        dataset_cfg = dataset_cfg.get('dataset')
        # dataset_cfg['data_root'] = '../'+dataset_cfg['data_root']
        dataset = build_dataset(dataset_cfg)
        cache_datasets.append(dataset)

    # ctx = torch.multiprocessing.get_context("spawn")
    num_workers = 8
    for idx, dataset in enumerate(cache_datasets):
        all_items = []
        pbar = ProgressBar(len(dataset))
        for i, item in zip(range(len(dataset)), dataset):
            pbar.update()
            all_items.append(item)

        num_p_list = []
        slice_len = len(all_items) // num_workers
        for local_rank_id in range(num_workers):
            if is_cuda:
                device = f'cuda:{local_rank_id}'
            else:
                device = 'cpu'
            model = init_model(cfg, device=device)
            slice_items = all_items[local_rank_id * slice_len: (local_rank_id + 1) * slice_len]
            if local_rank_id == num_workers - 1:
                slice_items = all_items[local_rank_id * slice_len:]
            # model_forward_save(slice_items, model, args.output_dir, phases[idx], device, local_rank_id)
            p = torch.multiprocessing.Process(target=model_forward_save, args=(slice_items, model, args.output_dir, phases[idx], device, local_rank_id))
            p.start()
            num_p_list.append(p)
        for p in num_p_list:
            p.join()


if __name__ == '__main__':
    main()

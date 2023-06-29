import argparse
import os.path as osp
import sys
sys.path.insert(0, sys.path[0]+'/../..')
import mmengine
import torch

from torch.nn import functional as F
from mmengine.config import Config, DictAction
from mmengine.utils import ProgressBar

from mmpl.datasets.builder import build_dataset
from mmpl.utils import register_all_modules


def parse_args():
    parser = argparse.ArgumentParser(description='Browse a dataset')
    parser.add_argument('--config', default='configs/seg/seg_just_backbone_with_clip_config.py', help='train config file path')
    parser.add_argument(
        '--output-dir',
        '-o',
        default='cache_data/clip_data',
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


def init_clip_model(clip_config, device='cuda:0'):
    from transformers import AutoProcessor, CLIPModel, AutoTokenizer
    model = CLIPModel.from_pretrained(clip_config).to(device)
    model = model.eval()
    tokenizer = AutoTokenizer.from_pretrained(clip_config)
    inputs = tokenizer("a photo of a building", return_tensors="pt")
    for k, v in inputs.items():
        inputs[k] = v.to(device)
    text_features = model.get_text_features(**inputs).detach()  # 1, 512
    processor = AutoProcessor.from_pretrained(clip_config)
    size = (processor.image_processor.crop_size['width'], processor.image_processor.crop_size['height'])
    mean = torch.tensor(processor.image_processor.image_mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(processor.image_processor.image_std, device=device).view(1, 3, 1, 1)
    return model, tokenizer, text_features, size, mean, std

def model_forward(results, model, tokenizer, text_features, size, mean, std, device='cuda:0'):
    image = results['inputs'].unsqueeze(0).clone().detach().float().to(device)
    image = F.interpolate(image, size=size, mode='bicubic', align_corners=False)
    image = image / 255.
    image = (image - mean) / std
    image = image[:, [2, 1, 0], :, :]
    pixel_values = image
    vision_outputs = model.vision_model(pixel_values=pixel_values)
    img_dense_embs = vision_outputs['last_hidden_state'][:, 1:, :]
    img_dense_embs = model.visual_projection(img_dense_embs)
    img_dense_embs = img_dense_embs / img_dense_embs.norm(p=2, dim=-1, keepdim=True)
    text_embeds = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    # cosine similarity as logits
    logit_scale = model.logit_scale.exp()
    logits_per_image = torch.matmul(img_dense_embs, text_embeds.t()) * logit_scale
    return {'img_dense_embs': img_dense_embs.cpu(), 'logits_per_image': logits_per_image.cpu()}

def main():
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

    clip_config_ = 'pretrain/clip/models--openai--clip-vit-large-patch14-336/blobs'

    model, tokenizer, text_features, size, mean, std = init_clip_model(clip_config_)

    cache_datasets = []
    for phase in phases:
        dataset_cfg = cfg.get('datamodule_cfg').get(phase + '_loader', None)
        if dataset_cfg is None:
            continue
        dataset_cfg = dataset_cfg.get('dataset')
        # dataset_cfg['data_root'] = '../'+dataset_cfg['data_root']
        dataset = build_dataset(dataset_cfg)
        cache_datasets.append(dataset)
    for idx, dataset in enumerate(cache_datasets):
        progress_bar = ProgressBar(len(dataset))
        for i, item in zip(range(len(dataset)), dataset):
            progress_bar.update()
            cache_data = model_forward(item, model, tokenizer, text_features, size, mean, std)
            img_path = item['data_samples'].img_path
            mmengine.dump(cache_data, f"{args.output_dir}/{phases[idx]}_{osp.splitext(osp.basename(img_path))[0]}.pkl")


if __name__ == '__main__':
    main()

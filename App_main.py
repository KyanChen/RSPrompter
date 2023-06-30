import glob
import mmcv
import mmengine
import numpy as np
import os
from mmengine import Config, get
from mmengine.dataset import Compose
from mmpl.registry import MODELS, VISUALIZERS
from mmpl.utils import register_all_modules
register_all_modules()
# os.system('nvidia-smi')
# os.system('ls /usr/local')
# os.system('pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu117')
# os.system('pip install -U openmim')
# os.system('mim install mmcv==2.0.0')
# os.system('mim install mmengine')

import gradio as gr
import torch

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def construct_sample(img, pipeline):
    img = np.array(img)[:, :, ::-1]
    inputs = {
        'ori_shape': img.shape[:2],
        'img': img,
    }
    pipeline = Compose(pipeline)
    sample = pipeline(inputs)
    return sample

def build_model(cp, model_cfg):
    model_cpkt = torch.load(cp, map_location='cpu')
    model = MODELS.build(model_cfg)
    model.load_state_dict(model_cpkt, strict=True)
    model.to(device=device)
    model.eval()
    return model


# Function for building extraction
def inference_func(ori_img, cp):
    checkpoint = f'pretrain/{cp}_anchor.pth'
    cfg = f'configs/huggingface/rsprompter_anchor_{cp}_config.py'
    cfg = Config.fromfile(cfg)
    sample = construct_sample(ori_img, cfg.predict_pipeline)
    sample['inputs'] = [sample['inputs']]
    sample['data_samples'] = [sample['data_samples']]

    print('Use: ', device)
    model = build_model(checkpoint, cfg.model_cfg)

    with torch.no_grad():
        pred_results = model.predict_step(sample, batch_idx=0)

    cfg.visualizer.setdefault('save_dir', 'visualizer')
    visualizer = VISUALIZERS.build(cfg.visualizer)

    data_sample = pred_results[0]
    img = np.array(ori_img).copy()
    out_file = 'visualizer/test_img.jpg'
    mmengine.mkdir_or_exist(os.path.dirname(out_file))
    visualizer.add_datasample(
        'test_img',
        img,
        draw_gt=False,
        data_sample=data_sample,
        show=False,
        wait_time=0.01,
        pred_score_thr=0.4,
        out_file=out_file
    )
    img_bytes = get(out_file)
    img = mmcv.imfrombytes(img_bytes, channel_order='rgb')
    return img

title = "RSPrompter"
description = "Gradio demo for RSPrompter. Upload image from WHU building dataset, NWPU dataset, or SSDD Dataset or click any one of the examples, " \
              "Then select the prompt model, and click \"Submit\" and wait for the result. \n \n" \
              "Paper: RSPrompter: Learning to Prompt for Remote Sensing Instance Segmentation based on Visual Foundation Model"

article = "<p style='text-align: center'><a href='https://kyanchen.github.io/RSPrompter/' target='_blank'>RSPrompter Project " \
          "Page</a></p> "

files = glob.glob('examples/NWPU*')
examples = [[f, f.split('/')[-1].split('_')[0]] for f in files]

with gr.Blocks() as demo:
    image_input = gr.Image(type='pil', label='Input Img')
    # with gr.Row().style(equal_height=True):
    # image_LR_output = gr.outputs.Image(label='LR Img', type='numpy')
    image_output = gr.Image(label='Segment Result', type='numpy')
    with gr.Row():
        checkpoint = gr.Radio(['WHU', 'NWPU', 'SSDD'], label='Checkpoint')

io = gr.Interface(fn=inference_func,
                  inputs=[image_input, checkpoint],
                  outputs=[image_output],
                  title=title,
                  description=description,
                  article=article,
                  allow_flagging='auto',
                  examples=examples,
                  cache_examples=True,
                  layout="grid"
                  )
io.launch()

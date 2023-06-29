# RSPrompter: Learning to Prompt for Remote Sensing Instance Segmentation based on Visual Foundation Model

English | [简体中文](/readme_cn.md)

This is the pytorch implement of our paper "RSPrompter: Learning to Prompt for Remote Sensing Instance Segmentation based on Visual Foundation Model"


[Project Page](https://kyanchen.github.io/RSPrompter/) $\cdot$ [PDF Download](https://arxiv.org/abs/2306.16269) $\cdot$ [HuggingFace Demo](https://huggingface.co/spaces/KyanChen/RSPrompter)


## 0. Environment Setup

### 0.1 Create a virtual environment

```shell
conda create -n RSPrompter python=3.10
```

### 0.2 Activate the virtual environment
```sehll
conda activate RSPrompter
```

### 0.3 Install pytorch
Version of 1.x is also work, but the version of 2.x is recommended.
```shell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117
```

### 0.3 [Optional] Install pytorch
```shell
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

### 0.4 Install mmcv
Version of 2.x is recommended.
```shell
pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu117/torch2.0/index.html
```
Please refer to [installation documentation](https://mmcv.readthedocs.io/en/latest/get_started/installation.html) for more detailed installation.

### 0.5 Install other dependencies
```shell
pip install -r requirements.txt
```

If you find this project useful for your research, please cite our paper.

If you have any other questions, please contact me!!!
    
```
@misc{chen2023rsprompter,
      title={RSPrompter: Learning to Prompt for Remote Sensing Instance Segmentation based on Visual Foundation Model}, 
      author={Keyan Chen and Chenyang Liu and Hao Chen and Haotian Zhang and Wenyuan Li and Zhengxia Zou and Zhenwei Shi},
      year={2023},
      eprint={2306.16269},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
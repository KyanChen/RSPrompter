# RSPrompter: Learning to Prompt for Remote Sensing Instance Segmentation based on Visual Foundation Model

<div align="center">
<img src="https://github.com/KyanChen/RSPrompter/blob/cky/logo.png" width="60%">
</div>

English | [简体中文](/readme_cn.md)

This is the pytorch implement of our paper "RSPrompter: Learning to Prompt for Remote Sensing Instance Segmentation based on Visual Foundation Model"

This method will be integrated into the MMdetection framework soon, please stand by. If this work is helpful to you, please **star** this repository.

[Project Page](https://kyanchen.github.io/RSPrompter/) $\cdot$ [PDF Download](https://arxiv.org/abs/2306.16269) $\cdot$ [![Paper page](paper-page-sm-dark.svg)](https://huggingface.co/papers/2306.16269) $\cdot$ [![Open in Spaces](open-in-hf-spaces-sm-dark.svg)](https://huggingface.co/spaces/KyanChen/RSPrompter)


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

## 1. Data Preparation

### 1.1 Dataset

#### WHU Dataset
WHU dataset can be downloaded from [WHU](https://aistudio.baidu.com/aistudio/datasetdetail/56502). After downloading, put the dataset into the **data** folder, which contains some image examples.

#### NWPU Dataset
NWPU dataset can be downloaded from [NWPU](https://aistudio.baidu.com/aistudio/datasetdetail/52812). After downloading, put the dataset into the **data** folder, which contains some image examples.

#### SSDD Dataset
SSDD dataset can be downloaded from [SSDD](https://aistudio.baidu.com/aistudio/datasetdetail/100924). After downloading, put the dataset into the **data** folder, which contains some image examples.

#### 1.2 Split the dataset into train and test set
The dataset split files and annotation files are provided in this project, which are stored in the **data/*/annotations** folder in COCO annotation format.

## 2. Model Training

### 2.1 Train SAM-based model

#### 2.1.1 Config file
The config file is located in the **configs/rsprompter** folder, which can be modified according to the situation. The config file provides three models: SAM-seg, SAM-det, and RSPrompter.

#### 2.1.2 Train
Some parameters of the training can also be modified in the above configuration file. The main modification of the parameters in trainer_cfg, such as single-card multi-card training, etc., for specific configuration modifications, please refer to the Trainer of Pytorch Lightning.
```shell
python tools/train.py
```

### 2.2 [Optional] Train other models
#### 2.2.1 Config file
The config file is located in the **configs/rsprompter** folder, which provides only the configuration of Mask R-CNN and Mask2Former. The configuration of other models can refer to these two configuration files and the model config in MMDetection.

#### 2.2.2 Train
Modify the config path in **tools/train.py** and then run
```shell
python tools/train.py
```

## 3. Model Evaluation
The config file is located in the **configs/rsprompter** folder, which can be modified according to the situation.
When the val_evaluator and val_loader are configured in the configuration file, the model will automatically evaluate the model on the validation set during model training, and the evaluation results will be uploaded to Wandb and can be viewed in Wandb.
If you need to perform offline evaluation on the test set, you need to configure the test_evaluator and test_loader in the configuration file, as well as the config and ckpt-path paths in **tools/test.py**, and then run
```shell
python tools/test.py
```

## 4. [Optional] Model Visualization
The config file is located in the **configs/rsprompter** folder, which can be modified according to the situation. You can modify the parameters of DetVisualizationHook and DetLocalVisualizer in the configuration file, as well as the config and ckpt-path paths in **tools/predict.py**, and then run
```shell
python tools/predict.py
```

## 5. [Optional] Model Download
This project provides the model weights of RSPrompter-anchor, which are located in [huggingface space](https://huggingface.co/spaces/KyanChen/RSPrompter/tree/main/pretrain)

## 6. [Optional] Citation
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

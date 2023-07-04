# RSPrompter: Learning to Prompt for Remote Sensing Instance Segmentation based on Visual Foundation Model

[English](/readme.md) | 简体中文


本项目是论文"RSPrompter: Learning to Prompt for Remote Sensing Instance Segmentation based on Visual Foundation Model"的Pytorch实现


[项目主页](https://kyanchen.github.io/RSPrompter/) $\cdot$ [PDF下载](https://arxiv.org/abs/2306.16269) $\cdot$ [![论文页面](paper-page-sm-dark.svg)](https://huggingface.co/papers/2306.16269) $\cdot$ [![HuggingFace Space](open-in-hf-spaces-sm-dark.svg)](https://huggingface.co/spaces/KyanChen/RSPrompter)


## 0. 环境准备
### 0.1 建立虚拟环境
```shell
conda create -n RSPrompter python=3.10
```
### 0.2 激活虚拟环境
```sehll
conda activate RSPrompter
```
### 0.3 安装pytorch
1.x版本也可以，但是建议使用2.x版本
```shell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117
```
### 0.3 [可选]安装pytorch
```shell
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```
### 0.4 安装mmcv
建议2.x版本
```shell
pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu117/torch2.0/index.html
```
更多安装信息请参考[安装文档](https://mmcv.readthedocs.io/zh_CN/latest/get_started/installation.html)
### 0.5 安装其他依赖
```shell
pip install -r requirements.txt
```

## 1. 数据准备

### 1.1 数据集

#### WHU数据集
WHU数据集可以从[WHU](https://aistudio.baidu.com/aistudio/datasetdetail/56502)下载，下载后将数据集放到**data**文件夹中，该文件夹放入了一些图像示例。

#### NWPU数据集
NWPU数据集可以从[NWPU](https://aistudio.baidu.com/aistudio/datasetdetail/52812)下载，下载后将数据集放到**data**文件夹中，该文件夹放入了一些图像示例。

#### SSDD数据集
SSDD数据集可以从[SSDD](https://aistudio.baidu.com/aistudio/datasetdetail/100924)下载，下载后将数据集放到**data**文件夹中，该文件夹放入了一些图像示例。

### 1.2 划分训练测试集
在本项目中已提供论文中的数据集划分文件和标注文件，以COCO标注格式存储，位于**data/*/annotations**文件夹中。

## 2. 模型训练

### 2.1 训练SAM-based模型

#### 2.1.1 配置文件
配置文件位于**configs/rsprompter**文件夹中，可以依据情况修改该文件中的参数，提供了SAM-seg，SAM-det，RSPrompter三种模型的配置文件。

#### 2.1.2 训练
训练的一些参数配置也可以在上述配置文件中修改，主要修改trainer_cfg中的参数，例如单卡多卡训练等，具体配置修改参考Pytorch Lightning的Trainer。
```shell
python tools/train.py
```


### 2.2 [可选] 训练其他模型
#### 2.2.1 配置文件
配置文件位于**configs/rsprompter**文件夹中，仅提供了Mask R-CNN和Mask2Former的配置，其他模型的配置可以参考这两个配置文件和MMDetection中的模型config进行修改。

#### 2.2.2 训练
修改**tools/train.py**中的config路径，然后运行
```shell
python tools/train.py
```


## 3. 模型评测

模型配置文件位于**configs/rsprompter**文件夹中，可以依据情况修改该文件中的参数。
当配置了该文件中val_evaluator和val_loader，在模型训练时，会自动进行模型在验证集上的评测，评测结果会上传到Wandb中，可以在Wandb中查看。
如果需要在测试集上进行离线评测，需要配置配置文件中的test_evaluator和test_loader，以及**tools/test.py**中的config和ckpt-path路径，然后运行
```shell
python tools/test.py
```

## 4. [可选]结果可视化
模型配置文件位于**configs/rsprompter**文件夹中，可以依据情况修改该文件中的**DetVisualizationHook**和**DetLocalVisualizer**的参数，
以及**tools/predict.py**中的config和ckpt-path路径，然后运行
```shell
python tools/predict.py
```


## 5. [可选]模型下载
本项目提供了RSPrompter-anchor的模型权重，位于[huggingface space](https://huggingface.co/spaces/KyanChen/RSPrompter/tree/main/pretrain)中

## 6. [可选]引用
如果您认为本项目对您的研究有所帮助，请引用我们的论文.

如果您有其他问题，请联系我！！！

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

# Changelog v2.x

### v2.25.0 (31/5/2022)

#### Highlights

- Support dedicated `WandbLogger` hook
- Support [ConvNeXt](configs/convnext), [DDOD](configs/ddod), [SOLOv2](configs/solov2)
- Support [Mask2Former](configs/mask2former) for instance segmentation
- Rename [config files of Mask2Former](configs/mask2former)

#### Backwards incompatible changes

- Rename [config files of Mask2Former](configs/mask2former) (#7571)

  <table align="center">
    <thead>
        <tr align='center'>
            <td>before v2.25.0</td>
            <td>after v2.25.0</td>
        </tr>
    </thead>
    <tbody><tr valign='top'>
    <th>

  - `mask2former_xxx_coco.py` represents config files for **panoptic segmentation**.

  </th>
    <th>

  - `mask2former_xxx_coco.py` represents config files for **instance segmentation**.
  - `mask2former_xxx_coco-panoptic.py` represents config files for **panoptic segmentation**.

  </th></tr>
  </tbody></table>

#### New Features

- Support [ConvNeXt](https://arxiv.org/abs/2201.03545) (#7281)
- Support [DDOD](https://arxiv.org/abs/2107.02963) (#7279)
- Support [SOLOv2](https://arxiv.org/abs/2003.10152) (#7441)
- Support [Mask2Former](https://arxiv.org/abs/2112.01527) for instance segmentation (#7571, #8032)

#### Bug Fixes

- Enable YOLOX training on different devices (#7912)
- Fix the log plot error when evaluation with `interval != 1` (#7784)
- Fix RuntimeError of HTC (#8083)

#### Improvements

- Support dedicated `WandbLogger` hook (#7459)

  Users can set

  ```python
  cfg.log_config.hooks = [
    dict(type='MMDetWandbHook',
         init_kwargs={'project': 'MMDetection-tutorial'},
         interval=10,
         log_checkpoint=True,
         log_checkpoint_metadata=True,
         num_eval_images=10)]
  ```

  in the config to use `MMDetWandbHook`. Example can be found in this [colab tutorial](https://colab.research.google.com/drive/1RCSXHZwDZvakFh3eo9RuNrJbCGqD0dru?usp=sharing#scrollTo=WTEdPDRaBz2C)

- Add `AvoidOOM` to avoid OOM (#7434, #8091)

  Try to use `AvoidCUDAOOM` to avoid GPU out of memory. It will first retry after calling `torch.cuda.empty_cache()`. If it still fails, it will then retry by converting the type of inputs to FP16 format. If it still fails, it will try to copy inputs from GPUs to CPUs to continue computing. Try AvoidOOM in code to make the code continue to run when GPU memory runs out:

  ```python
  from mmdet.utils import AvoidCUDAOOM

  output = AvoidCUDAOOM.retry_if_cuda_oom(some_function)(input1, input2)
  ```

  Users can also try `AvoidCUDAOOM` as a decorator to make the code continue to run when GPU memory runs out:

  ```python
  from mmdet.utils import AvoidCUDAOOM

  @AvoidCUDAOOM.retry_if_cuda_oom
  def function(*args, **kwargs):
      ...
      return xxx
  ```

- Support reading `gpu_collect` from `cfg.evaluation.gpu_collect` (#7672)

- Speedup the Video Inference by Accelerating data-loading Stage (#7832)

- Support replacing the `${key}` with the value of `cfg.key` (#7492)

- Accelerate result analysis in `analyze_result.py`. The evaluation time is speedup by 10 ~ 15 times and only tasks 10 ~ 15 minutes now. (#7891)

- Support to set `block_dilations` in `DilatedEncoder` (#7812)

- Support panoptic segmentation result analysis (#7922)

- Release DyHead with Swin-Large backbone (#7733)

- Documentations updating and adding

  - Fix wrong default type of `act_cfg` in `SwinTransformer` (#7794)
  - Fix text errors in the tutorials (#7959)
  - Rewrite the [installation guide](docs/en/get_started.md) (#7897)
  - [Useful hooks](docs/en/tutorials/useful_hooks.md) (#7810)
  - Fix heading anchor in documentation  (#8006)
  - Replace `markdownlint` with `mdformat` for avoiding installing ruby (#8009)

#### Contributors

A total of 20 developers contributed to this release.

Thanks @ZwwWayne, @DarthThomas, @solyaH, @LutingWang, @chenxinfeng4, @Czm369, @Chenastron, @chhluo, @austinmw, @Shanyaliux @hellock, @Y-M-Y, @jbwang1997, @hhaAndroid, @Irvingao, @zhanggefan, @BIGWangYuDong, @Keiku, @PeterVennerstrom, @ayulockin

### v2.24.0 (26/4/2022)

#### Highlights

- Support [Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation](https://arxiv.org/abs/2012.07177)
- Support automatically scaling LR according to GPU number and samples per GPU
- Support Class Aware Sampler that improves performance on OpenImages Dataset

#### New Features

- Support [Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation](https://arxiv.org/abs/2012.07177), see [example configs](configs/simple_copy_paste/mask_rcnn_r50_fpn_syncbn-all_rpn-2conv_ssj_scp_32x2_270k_coco.py) (#7501)

- Support Class Aware Sampler, users can set

  ```python
  data=dict(train_dataloader=dict(class_aware_sampler=dict(num_sample_class=1))))
  ```

  in the config to use `ClassAwareSampler`. Examples can be found in [the configs of OpenImages Dataset](https://github.com/open-mmlab/mmdetection/tree/main/configs/openimages/faster_rcnn_r50_fpn_32x2_cas_1x_openimages.py).  (#7436)

- Support automatically scaling LR according to GPU number and samples per GPU. (#7482)
  In each config, there is a corresponding config of auto-scaling LR as below,

  ```python
  auto_scale_lr = dict(enable=True, base_batch_size=N)
  ```

  where `N` is the batch size used for the current learning rate in the config (also equals to `samples_per_gpu` * gpu number to train this config).
  By default, we set `enable=False` so that the original usages will not be affected. Users can set `enable=True` in each config or add `--auto-scale-lr` after the command line to enable this feature and should check the correctness of `base_batch_size` in customized configs.

- Support setting dataloader arguments in config and add functions to handle config compatibility. (#7668)
  The comparison between the old and new usages is as below.

  <table align="center">
    <thead>
        <tr align='center'>
            <td>v2.23.0</td>
            <td>v2.24.0</td>
        </tr>
    </thead>
    <tbody><tr valign='top'>
    <th>

  ```python
  data = dict(
      samples_per_gpu=64, workers_per_gpu=4,
      train=dict(type='xxx', ...),
      val=dict(type='xxx', samples_per_gpu=4, ...),
      test=dict(type='xxx', ...),
  )
  ```

  </th>
    <th>

  ```python
  # A recommended config that is clear
  data = dict(
      train=dict(type='xxx', ...),
      val=dict(type='xxx', ...),
      test=dict(type='xxx', ...),
      # Use different batch size during inference.
      train_dataloader=dict(samples_per_gpu=64, workers_per_gpu=4),
      val_dataloader=dict(samples_per_gpu=8, workers_per_gpu=2),
      test_dataloader=dict(samples_per_gpu=8, workers_per_gpu=2),
  )

  # Old style still works but allows to set more arguments about data loaders
  data = dict(
      samples_per_gpu=64,  # only works for train_dataloader
      workers_per_gpu=4,  # only works for train_dataloader
      train=dict(type='xxx', ...),
      val=dict(type='xxx', ...),
      test=dict(type='xxx', ...),
      # Use different batch size during inference.
      val_dataloader=dict(samples_per_gpu=8, workers_per_gpu=2),
      test_dataloader=dict(samples_per_gpu=8, workers_per_gpu=2),
  )
  ```

  </th></tr>
  </tbody></table>

- Support memory profile hook. Users can use it to monitor the memory usages during training as below (#7560)

  ```python
  custom_hooks = [
      dict(type='MemoryProfilerHook', interval=50)
  ]
  ```

- Support to run on PyTorch with MLU chip (#7578)

- Support re-spliting data batch with tag (#7641)

- Support the `DiceCost` used by [K-Net](https://arxiv.org/abs/2106.14855) in `MaskHungarianAssigner` (#7716)

- Support splitting COCO data for Semi-supervised object detection (#7431)

- Support Pathlib for Config.fromfile (#7685)

- Support to use file client in OpenImages dataset (#7433)

- Add a probability parameter to Mosaic transformation (#7371)

- Support specifying interpolation mode in `Resize` pipeline (#7585)

#### Bug Fixes

- Avoid invalid bbox after deform_sampling (#7567)
- Fix the issue that argument color_theme does not take effect when exporting confusion matrix (#7701)
- Fix the `end_level` in Necks, which should be the index of the end input backbone level (#7502)
- Fix the bug that `mix_results` may be None in `MultiImageMixDataset` (#7530)
- Fix the bug in ResNet plugin when two plugins are used (#7797)

#### Improvements

- Enhance `load_json_logs` of analyze_logs.py for resumed training logs (#7732)
- Add argument `out_file` in image_demo.py (#7676)
- Allow mixed precision training with `SimOTAAssigner` (#7516)
- Updated INF to 100000.0 to be the same as that in the official YOLOX (#7778)
- Add documentations of:
  - how to get channels of a new backbone (#7642)
  - how to unfreeze the backbone network (#7570)
  - how to train fast_rcnn model (#7549)
  - proposals in Deformable DETR (#7690)
  - from-scratch install script in get_started.md (#7575)
- Release pre-trained models of
  - [Mask2Former](configs/mask2former) (#7595, #7709)
  - RetinaNet with ResNet-18 and release models (#7387)
  - RetinaNet with EfficientNet backbone (#7646)

#### Contributors

A total of 27 developers contributed to this release.
Thanks @jovialio, @zhangsanfeng2022, @HarryZJ, @jamiechoi1995, @nestiank, @PeterH0323, @RangeKing, @Y-M-Y, @mattcasey02, @weiji14, @Yulv-git, @xiefeifeihu, @FANG-MING, @meng976537406, @nijkah, @sudz123, @CCODING04, @SheffieldCao, @Czm369, @BIGWangYuDong, @zytx121, @jbwang1997, @chhluo, @jshilong, @RangiLyu, @hhaAndroid, @ZwwWayne

### v2.23.0 (28/3/2022)

#### Highlights

- Support Mask2Former: [Masked-attention Mask Transformer for Universal Image Segmentation](https://arxiv.org/abs/2112.01527)
- Support EfficientNet: [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
- Support setting data root through environment variable `MMDET_DATASETS`, users don't have to modify the corresponding path in config files anymore.
- Find a good recipe for fine-tuning high precision ResNet backbone pre-trained by Torchvision.

#### New Features

- Support [Mask2Former](configs/mask2former)(#6938)(#7466)(#7471)
- Support [EfficientNet](configs/efficientnet) (#7514)
- Support setting data root through environment variable `MMDET_DATASETS`, users don't have to modify the corresponding path in config files anymore. (#7386)
- Support setting different seeds to different ranks (#7432)
- Update the `dist_train.sh` so that the script can be used to support launching multi-node training on machines without slurm (#7415)
- Find a good recipe for fine-tuning high precision ResNet backbone pre-trained by Torchvision (#7489)

#### Bug Fixes

- Fix bug in VOC unit test which removes the data directory (#7270)
- Adjust the order of `get_classes` and `FileClient` (#7276)
- Force the inputs of `get_bboxes` in yolox_head to float32 (#7324)
- Fix misplaced arguments in LoadPanopticAnnotations (#7388)
- Fix reduction=mean in CELoss. (#7449)
- Update unit test of CrossEntropyCost (#7537)
- Fix memory leaking in panpotic segmentation evaluation (#7538)
- Fix the bug of shape broadcast in YOLOv3 (#7551)

#### Improvements

- Add Chinese version of onnx2tensorrt.md (#7219)
- Update colab tutorials (#7310)
- Update information about Localization Distillation (#7350)
- Add Chinese version of `finetune.md` (#7178)
- Update YOLOX log for non square input (#7235)
- Add `nproc` in `coco_panoptic.py` for panoptic quality computing (#7315)
- Allow to set channel_order in LoadImageFromFile (#7258)
- Take point sample related functions out of mask_point_head (#7353)
- Add instance evaluation for coco_panoptic (#7313)
- Enhance the robustness of analyze_logs.py (#7407)
- Supplementary notes of sync_random_seed (#7440)
- Update docstring of cross entropy loss (#7472)
- Update pascal voc result (#7503)
- We create How-to documentation to record any questions about How to xxx. In this version, we added
  - How to use Mosaic augmentation (#7507)
  - How to use backbone in mmcls (#7438)
  - How to produce and submit the prediction results of panoptic segmentation models on COCO test-dev set (#7430))

#### Contributors

A total of 27 developers contributed to this release.
Thanks @ZwwWayne, @haofanwang, @shinya7y, @chhluo, @yangrisheng, @triple-Mu, @jbwang1997, @HikariTJU, @imflash217, @274869388, @zytx121, @matrixgame2018, @jamiechoi1995, @BIGWangYuDong, @JingweiZhang12, @Xiangxu-0103, @hhaAndroid, @jshilong, @osbm, @ceroytres, @bunge-bedstraw-herb, @Youth-Got, @daavoo, @jiangyitong, @RangiLyu, @CCODING04, @yarkable

### v2.22.0 (24/2/2022)

#### Highlights

- Support MaskFormer: [Per-Pixel Classification is Not All You Need for Semantic Segmentation](https://arxiv.org/abs/2107.06278) (#7212)
- Support DyHead: [Dynamic Head: Unifying Object Detection Heads with Attentions](https://arxiv.org/abs/2106.08322) (#6823)
- Release a good recipe of using ResNet in object detectors pre-trained by [ResNet Strikes Back](https://arxiv.org/abs/2110.00476), which consistently brings about 3~4 mAP improvements over RetinaNet, Faster/Mask/Cascade Mask R-CNN (#7001)
- Support [Open Images Dataset](https://storage.googleapis.com/openimages/web/index.html) (#6331)
- Support TIMM backbone: [PyTorch Image Models](https://github.com/rwightman/pytorch-image-models) (#7020)

#### New Features

- Support [MaskFormer](configs/maskformer) (#7212)
- Support [DyHead](configs/dyhead) (#6823)
- Support [ResNet Strikes Back](configs/resnet_strikes_back) (#7001)
- Support [OpenImages Dataset](configs/openimages) (#6331)
- Support [TIMM backbone](configs/timm_example) (#7020)
- Support visualization for Panoptic Segmentation (#7041)

#### Breaking Changes

In order to support the visualization for Panoptic Segmentation, the `num_classes` can not be `None` when using the `get_palette` function to determine whether to use the panoptic palette.

#### Bug Fixes

- Fix bug for the best checkpoints can not be saved when the `key_score` is None (#7101)
- Fix MixUp transform filter boxes failing case (#7080)
- Add missing properties in SABLHead (#7091)
- Fix bug when NaNs exist in confusion matrix (#7147)
- Fix PALETTE AttributeError in downstream task (#7230)

#### Improvements

- Speed up SimOTA matching (#7098)
- Add Chinese translation of `docs_zh-CN/tutorials/init_cfg.md` (#7188)

#### Contributors

A total of 20 developers contributed to this release.
Thanks @ZwwWayne, @hhaAndroid, @RangiLyu, @AronLin, @BIGWangYuDong, @jbwang1997, @zytx121, @chhluo, @shinya7y, @LuooChen, @dvansa, @siatwangmin, @del-zhenwu, @vikashranjan26, @haofanwang, @jamiechoi1995, @HJoonKwon, @yarkable, @zhijian-liu, @RangeKing

### v2.21.0 (8/2/2022)

### Breaking Changes

To standardize the contents in config READMEs and meta files of OpenMMLab projects, the READMEs and meta files in each config directory have been significantly changed. The template will be released in the future, for now, you can refer to the examples of README for [algorithm](https://github.com/open-mmlab/mmdetection/blob/master/configs/faster_rcnn/README.md), [dataset](https://github.com/open-mmlab/mmdetection/blob/master/configs/deepfashion/README.md) and [backbone](https://github.com/open-mmlab/mmdetection/blob/master/configs/regnet/README.md). To align with the standard, the configs in dcn are put into to two directories named `dcn` and `dcnv2`.

#### New Features

- Allow to customize colors of different classes during visualization (#6716)
- Support CPU training (#7016)
- Add download script of COCO, LVIS, and VOC dataset (#7015)

#### Bug Fixes

- Fix weight conversion issue of RetinaNet with Swin-S (#6973)
- Update `__repr__` of `Compose` (#6951)
- Fix BadZipFile Error when build docker (#6966)
- Fix bug in non-distributed multi-gpu training/testing (#7019)
- Fix bbox clamp in PyTorch 1.10 (#7074)
- Relax the requirement of PALETTE in dataset wrappers (#7085)
- Keep the same weights before reassign in the PAA head (#7032)
- Update code demo in doc (#7092)

#### Improvements

- Speed-up training by allow to set variables of multi-processing (#6974, #7036)
- Add links of Chinese tutorials in readme (#6897)
- Disable cv2 multiprocessing by default for acceleration (#6867)
- Deprecate the support for "python setup.py test" (#6998)
- Re-organize metafiles and config readmes (#7051)
- Fix None grad problem during training TOOD by adding `SigmoidGeometricMean` (#7090)

#### Contributors

A total of 26 developers contributed to this release.
Thanks @del-zhenwu, @zimoqingfeng, @srishilesh, @imyhxy, @jenhaoyang, @jliu-ac, @kimnamu, @ShengliLiu, @garvan2021, @ciusji, @DIYer22, @kimnamu, @q3394101, @zhouzaida, @gaotongxiao, @topsy404, @AntoAndGar, @jbwang1997, @nijkah, @ZwwWayne, @Czm369, @jshilong, @RangiLyu, @BIGWangYuDong, @hhaAndroid, @AronLin

### v2.20.0 (27/12/2021)

#### New Features

- Support [TOOD](configs/tood/README.md): Task-aligned One-stage Object Detection (ICCV 2021 Oral) (#6746)
- Support resuming from the latest checkpoint automatically (#6727)

#### Bug Fixes

- Fix wrong bbox `loss_weight` of the PAA head (#6744)
- Fix the padding value of `gt_semantic_seg` in batch collating (#6837)
- Fix test error of lvis when using `classwise` (#6845)
- Avoid BC-breaking of `get_local_path`  (#6719)
- Fix bug in `sync_norm_hook` when the BN layer does not exist (#6852)
- Use pycocotools directly no matter what platform it is (#6838)

#### Improvements

- Add unit test for SimOTA with no valid bbox (#6770)
- Use precommit to check readme (#6802)
- Support selecting GPU-ids in non-distributed testing time (#6781)

#### Contributors

A total of 16 developers contributed to this release.
Thanks @ZwwWayne, @Czm369, @jshilong, @RangiLyu, @BIGWangYuDong, @hhaAndroid, @jamiechoi1995, @AronLin, @Keiku, @gkagkos, @fcakyon, @www516717402, @vansin, @zactodd, @kimnamu, @jenhaoyang

### v2.19.1 (14/12/2021)

#### New Features

- Release [YOLOX](configs/yolox/README.md) COCO pretrained models (#6698)

#### Bug Fixes

- Fix DCN initialization in DenseHead (#6625)
- Fix initialization of ConvFCHead (#6624)
- Fix PseudoSampler in RCNN (#6622)
- Fix weight initialization in Swin and PVT (#6663)
- Fix dtype bug in BaseDenseHead (#6767)
- Fix SimOTA with no valid bbox (#6733)

#### Improvements

- Add an example of combining swin and one-stage models (#6621)
- Add `get_ann_info` to dataset_wrappers (#6526)
- Support keeping image ratio in the multi-scale training of YOLOX (#6732)
- Support `bbox_clip_border` for the augmentations of YOLOX (#6730)

#### Documents

- Update metafile (#6717)
- Add mmhuman3d in readme (#6699)
- Update FAQ docs (#6587)
- Add doc for `detect_anomalous_params` (#6697)

#### Contributors

A total of 11 developers contributed to this release.
Thanks @ZwwWayne, @LJoson, @Czm369, @jshilong, @ZCMax, @RangiLyu, @BIGWangYuDong, @hhaAndroid, @zhaoxin111, @GT9505, @shinya7y

### v2.19.0 (29/11/2021)

#### Highlights

- Support [Label Assignment Distillation](https://arxiv.org/abs/2108.10520)
- Support `persistent_workers` for Pytorch >= 1.7
- Align accuracy to the updated official YOLOX

#### New Features

- Support [Label Assignment Distillation](https://arxiv.org/abs/2108.10520) (#6342)
- Support `persistent_workers` for Pytorch >= 1.7 (#6435)

#### Bug Fixes

- Fix repeatedly output warning message (#6584)
- Avoid infinite GPU waiting in dist training (#6501)
- Fix SSD512 config error (#6574)
- Fix MMDetection model to ONNX command (#6558)

#### Improvements

- Refactor configs of FP16 models (#6592)
- Align accuracy to the updated official YOLOX (#6443)
- Speed up training and reduce memory cost when using PhotoMetricDistortion. (#6442)
- Make OHEM work with seesaw loss (#6514)

#### Documents

- Update README.md (#6567)

#### Contributors

A total of 11 developers contributed to this release.
Thanks @FloydHsiu, @RangiLyu, @ZwwWayne, @AndreaPi, @st9007a, @hachreak, @BIGWangYuDong, @hhaAndroid, @AronLin, @chhluo, @vealocia, @HarborYuan, @st9007a, @jshilong

### v2.18.1 (15/11/2021)

#### Highlights

- Release [QueryInst](http://arxiv.org/abs/2105.01928) pre-trained weights (#6460)
- Support plot confusion matrix (#6344)

#### New Features

- Release [QueryInst](http://arxiv.org/abs/2105.01928) pre-trained weights (#6460)
- Support plot confusion matrix (#6344)

#### Bug Fixes

- Fix aug test error when the number of prediction bboxes is 0 (#6398)
- Fix SpatialReductionAttention in PVT (#6488)
- Fix wrong use of `trunc_normal_init` in PVT and Swin-Transformer (#6432)

#### Improvements

- Save the printed AP information of COCO API to logger (#6505)
- Always map location to cpu when load checkpoint (#6405)
- Set a random seed when the user does not set a seed (#6457)

#### Documents

- Chinese version of [Corruption Benchmarking](robustness_benchmarking.md) (#6375)
- Fix config path in docs (#6396)
- Update GRoIE readme (#6401)

#### Contributors

A total of 11 developers contributed to this release.
Thanks @st9007a, @hachreak, @HarborYuan, @vealocia, @chhluo, @AndreaPi, @AronLin, @BIGWangYuDong, @hhaAndroid, @RangiLyu, @ZwwWayne

### v2.18.0 (27/10/2021)

#### Highlights

- Support [QueryInst](http://arxiv.org/abs/2105.01928) (#6050)
- Refactor dense heads to decouple onnx export logics from `get_bboxes` and speed up inference (#5317, #6003, #6369, #6268, #6315)

#### New Features

- Support [QueryInst](http://arxiv.org/abs/2105.01928) (#6050)
- Support infinite sampler (#5996)

#### Bug Fixes

- Fix init_weight in fcn_mask_head (#6378)
- Fix type error in imshow_bboxes of RPN (#6386)
- Fix broken colab link in MMDetection Tutorial (#6382)
- Make sure the device and dtype of scale_factor are the same as bboxes (#6374)
- Remove sampling hardcode (#6317)
- Fix RandomAffine bbox coordinate recorrection (#6293)
- Fix init bug of final cls/reg layer in convfc head (#6279)
- Fix img_shape broken in auto_augment (#6259)
- Fix kwargs parameter missing error in two_stage (#6256)

#### Improvements

- Unify the interface of stuff head and panoptic head (#6308)
- Polish readme (#6243)
- Add code-spell pre-commit hook and fix a typo (#6306)
- Fix typo (#6245, #6190)
- Fix sampler unit test (#6284)
- Fix `forward_dummy` of YOLACT to enable `get_flops` (#6079)
- Fix link error in the config documentation (#6252)
- Adjust the order to beautify the document (#6195)

#### Refactors

- Refactor one-stage get_bboxes logic (#5317)
- Refactor ONNX export of One-Stage models (#6003, #6369)
- Refactor dense_head and speedup (#6268)
- Migrate to use prior_generator in training of dense heads (#6315)

#### Contributors

A total of 18 developers contributed to this release.
Thanks @Boyden, @onnkeat, @st9007a, @vealocia, @yhcao6, @DapangpangX, @yellowdolphin, @cclauss, @kennymckormick,
@pingguokiller, @collinzrj, @AndreaPi, @AronLin, @BIGWangYuDong, @hhaAndroid, @jshilong, @RangiLyu, @ZwwWayne

### v2.17.0 (28/9/2021)

#### Highlights

- Support [PVT](https://arxiv.org/abs/2102.12122) and [PVTv2](https://arxiv.org/abs/2106.13797)
- Support [SOLO](https://arxiv.org/abs/1912.04488)
- Support large scale jittering and New Mask R-CNN baselines
- Speed up `YOLOv3` inference

#### New Features

- Support [PVT](https://arxiv.org/abs/2102.12122) and [PVTv2](https://arxiv.org/abs/2106.13797) (#5780)
- Support [SOLO](https://arxiv.org/abs/1912.04488) (#5832)
- Support large scale jittering and New Mask R-CNN baselines (#6132)
- Add a general data structure for the results of models (#5508)
- Added a base class for one-stage instance segmentation (#5904)
- Speed up `YOLOv3` inference (#5991)
- Release Swin Transformer pre-trained models (#6100)
- Support mixed precision training in `YOLOX` (#5983)
- Support `val` workflow in `YOLACT` (#5986)
- Add script to test `torchserve` (#5936)
- Support `onnxsim` with dynamic input shape (#6117)

#### Bug Fixes

- Fix the function naming errors in `model_wrappers` (#5975)
- Fix regression loss bug when the input is an empty tensor (#5976)
- Fix scores not contiguous error in `centernet_head` (#6016)
- Fix missing parameters bug in `imshow_bboxes` (#6034)
- Fix bug in `aug_test` of `HTC` when the length of `det_bboxes` is 0 (#6088)
- Fix empty proposal errors in the training of some two-stage models (#5941)
- Fix `dynamic_axes` parameter error in `ONNX` dynamic shape export (#6104)
- Fix `dynamic_shape` bug of `SyncRandomSizeHook` (#6144)
- Fix the Swin Transformer config link error in the configuration (#6172)

#### Improvements

- Add filter rules in `Mosaic` transform (#5897)
- Add size divisor in get flops to avoid some potential bugs (#6076)
- Add Chinese translation of `docs_zh-CN/tutorials/customize_dataset.md` (#5915)
- Add Chinese translation of `conventions.md` (#5825)
- Add description of the output of data pipeline (#5886)
- Add dataset information in the README file for `PanopticFPN` (#5996)
- Add `extra_repr` for `DropBlock` layer to get details in the model printing (#6140)
- Fix CI out of memory and add PyTorch1.9 Python3.9 unit tests (#5862)
- Fix download links error of some model (#6069)
- Improve the generalization of XML dataset (#5943)
- Polish assertion error messages (#6017)
- Remove `opencv-python-headless` dependency by `albumentations` (#5868)
- Check dtype in transform unit tests (#5969)
- Replace the default theme of documentation with PyTorch Sphinx Theme (#6146)
- Update the paper and code fields in the metafile (#6043)
- Support to customize padding value of segmentation map (#6152)
- Support to resize multiple segmentation maps (#5747)

#### Contributors

A total of 24 developers contributed to this release.
Thanks @morkovka1337, @HarborYuan, @guillaumefrd, @guigarfr, @www516717402, @gaotongxiao, @ypwhs, @MartaYang, @shinya7y, @justiceeem, @zhaojinjian0000, @VVsssssk, @aravind-anantha, @wangbo-zhao, @czczup, @whai362, @czczup, @marijnl, @AronLin, @BIGWangYuDong, @hhaAndroid, @jshilong, @RangiLyu, @ZwwWayne

### v2.16.0 (30/8/2021)

#### Highlights

- Support [Panoptic FPN](https://arxiv.org/abs/1901.02446) and [Swin Transformer](https://arxiv.org/abs/2103.14030)

#### New Features

- Support [Panoptic FPN](https://arxiv.org/abs/1901.02446) and release models (#5577, #5902)
- Support Swin Transformer backbone (#5748)
- Release RetinaNet models pre-trained with multi-scale 3x schedule (#5636)
- Add script to convert unlabeled image list to coco format (#5643)
- Add hook to check whether the loss value is valid (#5674)
- Add YOLO anchor optimizing tool (#5644)
- Support export onnx models without post process. (#5851)
- Support classwise evaluation in CocoPanopticDataset (#5896)
- Adapt browse_dataset for concatenated datasets. (#5935)
- Add `PatchEmbed` and `PatchMerging` with `AdaptivePadding` (#5952)

#### Bug Fixes

- Fix unit tests of YOLOX (#5859)
- Fix lose randomness in `imshow_det_bboxes` (#5845)
- Make output result of `ImageToTensor` contiguous (#5756)
- Fix inference bug when calling `regress_by_class` in RoIHead in some cases (#5884)
- Fix bug in CIoU loss where alpha should not have gradient. (#5835)
- Fix the bug that `multiscale_output` is defined but not used in HRNet (#5887)
- Set the priority of EvalHook to LOW. (#5882)
- Fix a YOLOX bug when applying bbox rescaling in test mode (#5899)
- Fix mosaic coordinate error (#5947)
- Fix dtype of bbox in RandomAffine. (#5930)

#### Improvements

- Add Chinese version of `data_pipeline` and  (#5662)
- Support to remove state dicts of EMA when publishing models. (#5858)
- Refactor the loss function in HTC and SCNet (#5881)
- Use warnings instead of logger.warning (#5540)
- Use legacy coordinate in metric of VOC (#5627)
- Add Chinese version of customize_losses (#5826)
- Add Chinese version of model_zoo (#5827)

#### Contributors

A total of 19 developers contributed to this release.
Thanks @ypwhs, @zywvvd, @collinzrj, @OceanPang, @ddonatien, @@haotian-liu, @viibridges, @Muyun99, @guigarfr, @zhaojinjian0000, @jbwang1997,@wangbo-zhao, @xvjiarui, @RangiLyu, @jshilong, @AronLin, @BIGWangYuDong, @hhaAndroid, @ZwwWayne

### v2.15.1 (11/8/2021)

#### Highlights

- Support [YOLOX](https://arxiv.org/abs/2107.08430)

#### New Features

- Support [YOLOX](https://arxiv.org/abs/2107.08430)(#5756, #5758, #5760, #5767, #5770, #5774, #5777, #5808, #5828, #5848)

#### Bug Fixes

- Update correct SSD models. (#5789)
- Fix casting error in mask structure (#5820)
- Fix MMCV deployment documentation links. (#5790)

#### Improvements

- Use dynamic MMCV download link in TorchServe dockerfile (#5779)
- Rename the function `upsample_like` to `interpolate_as` for more general usage (#5788)

#### Contributors

A total of 14 developers contributed to this release.
Thanks @HAOCHENYE, @xiaohu2015, @HsLOL, @zhiqwang, @Adamdad, @shinya7y, @Johnson-Wang, @RangiLyu, @jshilong, @mmeendez8, @AronLin, @BIGWangYuDong, @hhaAndroid, @ZwwWayne

### v2.15.0 (02/8/2021)

#### Highlights

- Support adding [MIM](https://github.com/open-mmlab/mim) dependencies during pip installation
- Support MobileNetV2 for SSD-Lite and YOLOv3
- Support Chinese Documentation

#### New Features

- Add function `upsample_like` (#5732)
- Support to output pdf and epub format documentation (#5738)
- Support and release Cascade Mask R-CNN 3x pre-trained models (#5645)
- Add `ignore_index` to CrossEntropyLoss (#5646)
- Support adding [MIM](https://github.com/open-mmlab/mim) dependencies during pip installation (#5676)
- Add MobileNetV2 config and models for YOLOv3 (#5510)
- Support COCO Panoptic Dataset (#5231)
- Support ONNX export of cascade models (#5486)
- Support DropBlock with RetinaNet (#5544)
- Support MobileNetV2 SSD-Lite (#5526)

#### Bug Fixes

- Fix the device of label in multiclass_nms (#5673)
- Fix error of backbone initialization from pre-trained checkpoint in config file (#5603, #5550)
- Fix download links of RegNet pretrained weights (#5655)
- Fix two-stage runtime error given empty proposal (#5559)
- Fix flops count error in DETR (#5654)
- Fix unittest for `NumClassCheckHook` when it is not used. (#5626)
- Fix description bug of using custom dataset (#5546)
- Fix bug of `multiclass_nms` that returns the global indices (#5592)
- Fix `valid_mask` logic error in RPNHead (#5562)
- Fix unit test error of pretrained configs (#5561)
- Fix typo error in anchor_head.py (#5555)
- Fix bug when using dataset wrappers (#5552)
- Fix a typo error in demo/MMDet_Tutorial.ipynb (#5511)
- Fixing crash in `get_root_logger` when `cfg.log_level` is not None (#5521)
- Fix docker version (#5502)
- Fix optimizer parameter error when using `IterBasedRunner` (#5490)

#### Improvements

- Add unit tests for MMTracking (#5620)
- Add Chinese translation of documentation (#5718, #5618, #5558, #5423, #5593, #5421, #5408. #5369, #5419, #5530, #5531)
- Update resource limit (#5697)
- Update docstring for InstaBoost (#5640)
- Support key `reduction_override` in all loss functions (#5515)
- Use repeatdataset to accelerate CenterNet training (#5509)
- Remove unnecessary code in autoassign (#5519)
- Add documentation about `init_cfg` (#5273)

#### Contributors

A total of 18 developers contributed to this release.
Thanks @OceanPang, @AronLin, @hellock, @Outsider565, @RangiLyu, @ElectronicElephant, @likyoo, @BIGWangYuDong, @hhaAndroid, @noobying, @yyz561, @likyoo,
@zeakey, @ZwwWayne, @ChenyangLiu, @johnson-magic, @qingswu, @BuxianChen

### v2.14.0 (29/6/2021)

#### Highlights

- Add `simple_test` to dense heads to improve the consistency of single-stage and two-stage detectors
- Revert the `test_mixins` to single image test to improve efficiency and readability
- Add Faster R-CNN and Mask R-CNN config using multi-scale training with 3x schedule

#### New Features

- Support pretrained models from MoCo v2 and SwAV (#5286)
- Add Faster R-CNN and Mask R-CNN config using multi-scale training with 3x schedule (#5179, #5233)
- Add `reduction_override` in MSELoss (#5437)
- Stable support of exporting DETR to ONNX with dynamic shapes and batch inference (#5168)
- Stable support of exporting PointRend to ONNX with dynamic shapes and batch inference (#5440)

#### Bug Fixes

- Fix size mismatch bug in `multiclass_nms` (#4980)
- Fix the import path of `MultiScaleDeformableAttention` (#5338)
- Fix errors in config of GCNet ResNext101 models (#5360)
- Fix Grid-RCNN error when there is no bbox result (#5357)
- Fix errors in `onnx_export` of bbox_head when setting reg_class_agnostic (#5468)
- Fix type error of AutoAssign in the document (#5478)
- Fix web links ending with `.md` (#5315)

#### Improvements

- Add `simple_test` to dense heads to improve the consistency of single-stage and two-stage detectors (#5264)
- Add support for mask diagonal flip in TTA (#5403)
- Revert the `test_mixins` to single image test to improve efficiency and readability (#5249)
- Make YOLOv3 Neck more flexible (#5218)
- Refactor SSD to make it more general (#5291)
- Refactor `anchor_generator` and `point_generator` (#5349)
- Allow to configure out the `mask_head` of the HTC algorithm (#5389)
- Delete deprecated warning in FPN (#5311)
- Move `model.pretrained` to `model.backbone.init_cfg` (#5370)
- Make deployment tools more friendly to use (#5280)
- Clarify installation documentation (#5316)
- Add ImageNet Pretrained Models docs (#5268)
- Add FAQ about training loss=nan solution and COCO AP or AR =-1 (# 5312, #5313)
- Change all weight links of http to https (#5328)

### v2.13.0 (01/6/2021)

#### Highlights

- Support new methods: [CenterNet](https://arxiv.org/abs/1904.07850), [Seesaw Loss](https://arxiv.org/abs/2008.10032), [MobileNetV2](https://arxiv.org/abs/1801.04381)

#### New Features

- Support paper [Objects as Points](https://arxiv.org/abs/1904.07850) (#4602)
- Support paper [Seesaw Loss for Long-Tailed Instance Segmentation (CVPR 2021)](https://arxiv.org/abs/2008.10032) (#5128)
- Support [MobileNetV2](https://arxiv.org/abs/1801.04381) backbone and inverted residual block (#5122)
- Support [MIM](https://github.com/open-mmlab/mim) (#5143)
- ONNX exportation with dynamic shapes of CornerNet (#5136)
- Add `mask_soft` config option to allow non-binary masks (#4615)
- Add PWC metafile (#5135)

#### Bug Fixes

- Fix YOLOv3 FP16 training error (#5172)
- Fix Cacscade R-CNN TTA test error when `det_bboxes` length is 0  (#5221)
- Fix `iou_thr` variable naming errors in VOC recall calculation function (#5195)
- Fix Faster R-CNN performance dropped in ONNX Runtime (#5197)
- Fix DETR dict changed error when using python 3.8 during iteration  (#5226)

#### Improvements

- Refactor ONNX export of two stage detector (#5205)
- Replace MMDetection's EvalHook with MMCV's EvalHook for consistency  (#4806)
- Update RoI extractor for ONNX (#5194)
- Use better parameter initialization in YOLOv3 head for higher performance (#5181)
- Release new DCN models of Mask R-CNN by mixed-precision training (#5201)
- Update YOLOv3 model weights (#5229)
- Add DetectoRS ResNet-101 model weights (#4960)
- Discard bboxes with sizes equals to `min_bbox_size` (#5011)
- Remove duplicated code in DETR head (#5129)
- Remove unnecessary object in class definition (#5180)
- Fix doc link (#5192)

### v2.12.0 (01/5/2021)

#### Highlights

- Support new methods: [AutoAssign](https://arxiv.org/abs/2007.03496), [YOLOF](https://arxiv.org/abs/2103.09460), and [Deformable DETR](https://arxiv.org/abs/2010.04159)
- Stable support of exporting models to ONNX with batched images and dynamic shape (#5039)

#### Backwards Incompatible Changes

MMDetection is going through big refactoring for more general and convenient usages during the releases from v2.12.0 to v2.15.0 (maybe longer).
In v2.12.0 MMDetection inevitably brings some BC-breakings, including the MMCV dependency, model initialization, model registry, and mask AP evaluation.

- MMCV version. MMDetection v2.12.0 relies on the newest features in MMCV 1.3.3, including `BaseModule` for unified parameter initialization, model registry, and the CUDA operator `MultiScaleDeformableAttn` for [Deformable DETR](https://arxiv.org/abs/2010.04159). Note that MMCV 1.3.2 already contains all the features used by MMDet but has known issues. Therefore, we recommend users skip MMCV v1.3.2 and use v1.3.3, though v1.3.2 might work for most cases.
- Unified model initialization (#4750). To unify the parameter initialization in OpenMMLab projects, MMCV supports `BaseModule` that accepts `init_cfg` to allow the modules' parameters initialized in a flexible and unified manner. Now the users need to explicitly call `model.init_weights()` in the training script to initialize the model (as in [here](https://github.com/open-mmlab/mmdetection/blob/master/tools/train.py#L162), previously this was handled by the detector. The models in MMDetection have been re-benchmarked to ensure accuracy based on PR #4750. __The downstream projects should update their code accordingly to use MMDetection v2.12.0__.
- Unified model registry (#5059). To easily use backbones implemented in other OpenMMLab projects, MMDetection migrates to inherit the model registry created in MMCV (#760). In this way, as long as the backbone is supported in an OpenMMLab project and that project also uses the registry in MMCV, users can use that backbone in MMDetection by simply modifying the config without copying the code of that backbone into MMDetection.
- Mask AP evaluation (#4898). Previous versions calculate the areas of masks through the bounding boxes when calculating the mask AP of small, medium, and large instances. To indeed use the areas of masks, we pop the key `bbox` during mask AP calculation. This change does not affect the overall mask AP evaluation and aligns the mask AP of similar models in other projects like Detectron2.

#### New Features

- Support paper [AutoAssign: Differentiable Label Assignment for Dense Object Detection](https://arxiv.org/abs/2007.03496) (#4295)
- Support paper [You Only Look One-level Feature](https://arxiv.org/abs/2103.09460) (#4295)
- Support paper [Deformable DETR: Deformable Transformers for End-to-End Object Detection](https://arxiv.org/abs/2010.04159) (#4778)
- Support calculating IoU with FP16 tensor in `bbox_overlaps` to save memory and keep speed (#4889)
- Add `__repr__` in custom dataset to count the number of instances (#4756)
- Add windows support by updating requirements.txt (#5052)
- Stable support of exporting models to ONNX with batched images and dynamic shape, including SSD, FSAF,FCOS, YOLOv3, RetinaNet, Faster R-CNN, and Mask R-CNN (#5039)

#### Improvements

- Use MMCV `MODEL_REGISTRY` (#5059)
- Unified parameter initialization for more flexible usage (#4750)
- Rename variable names and fix docstring in anchor head (#4883)
- Support training with empty GT in Cascade RPN (#4928)
- Add more details of usage of `test_robustness` in documentation (#4917)
- Changing to use `pycocotools` instead of `mmpycocotools` to fully support Detectron2 and MMDetection in one environment (#4939)
- Update torch serve dockerfile to support dockers of more versions (#4954)
- Add check for training with single class dataset (#4973)
- Refactor transformer and DETR Head (#4763)
- Update FPG model zoo (#5079)
- More accurate mask AP of small/medium/large instances (#4898)

#### Bug Fixes

- Fix bug in mean_ap.py when calculating mAP by 11 points (#4875)
- Fix error when key `meta` is not in old checkpoints (#4936)
- Fix hanging bug when training with empty GT in VFNet, GFL, and FCOS by changing the place of `reduce_mean` (#4923, #4978, #5058)
- Fix asyncronized inference error and provide related demo (#4941)
- Fix IoU losses dimensionality unmatch error (#4982)
- Fix torch.randperm whtn using PyTorch 1.8 (#5014)
- Fix empty bbox error in `mask_head` when using CARAFE (#5062)
- Fix `supplement_mask` bug when there are zero-size RoIs (#5065)
- Fix testing with empty rois in RoI Heads (#5081)

### v2.11.0 (01/4/2021)

__Highlights__

- Support new method: [Localization Distillation for Object Detection](https://arxiv.org/pdf/2102.12252.pdf)
- Support Pytorch2ONNX with batch inference and dynamic shape

__New Features__

- Support [Localization Distillation for Object Detection](https://arxiv.org/pdf/2102.12252.pdf) (#4758)
- Support Pytorch2ONNX with batch inference and dynamic shape for Faster-RCNN and mainstream one-stage detectors (#4796)

__Improvements__

- Support batch inference in head of RetinaNet (#4699)
- Add batch dimension in second stage of Faster-RCNN (#4785)
- Support batch inference in bbox coder (#4721)
- Add check for `ann_ids` in `COCODataset` to ensure it is unique (#4789)
- support for showing the FPN results (#4716)
- support dynamic shape for grid_anchor (#4684)
- Move pycocotools version check to when it is used (#4880)

__Bug Fixes__

- Fix a bug of TridentNet when doing the batch inference (#4717)
- Fix a bug of Pytorch2ONNX in FASF (#4735)
- Fix a bug when show the image with float type (#4732)

### v2.10.0 (01/03/2021)

#### Highlights

- Support new methods: [FPG](https://arxiv.org/abs/2004.03580)
- Support ONNX2TensorRT for SSD, FSAF, FCOS, YOLOv3, and Faster R-CNN.

#### New Features

- Support ONNX2TensorRT for SSD, FSAF, FCOS, YOLOv3, and Faster R-CNN (#4569)
- Support [Feature Pyramid Grids (FPG)](https://arxiv.org/abs/2004.03580) (#4645)
- Support video demo (#4420)
- Add seed option for sampler (#4665)
- Support to customize type of runner (#4570, #4669)
- Support synchronizing BN buffer in `EvalHook` (#4582)
- Add script for GIF demo (#4573)

#### Bug Fixes

- Fix ConfigDict AttributeError and add Colab link (#4643)
- Avoid crash in empty gt training of GFL head (#4631)
- Fix `iou_thrs` bug in RPN evaluation (#4581)
- Fix syntax error of config when upgrading model version (#4584)

#### Improvements

- Refactor unit test file structures (#4600)
- Refactor nms config (#4636)
- Get loading pipeline by checking the class directly rather than through config strings (#4619)
- Add doctests for mask target generation and mask structures (#4614)
- Use deep copy when copying pipeline arguments (#4621)
- Update documentations (#4642, #4650, #4620, #4630)
- Remove redundant code calling `import_modules_from_strings` (#4601)
- Clean deprecated FP16 API (#4571)
- Check whether `CLASSES` is correctly initialized in the initialization of `XMLDataset` (#4555)
- Support batch inference in the inference API (#4462, #4526)
- Clean deprecated warning and fix 'meta' error (#4695)

### v2.9.0 (01/02/2021)

#### Highlights

- Support new methods: [SCNet](https://arxiv.org/abs/2012.10150), [Sparse R-CNN](https://arxiv.org/abs/2011.12450)
- Move `train_cfg` and `test_cfg` into model in configs
- Support to visualize results based on prediction quality

#### New Features

- Support [SCNet](https://arxiv.org/abs/2012.10150) (#4356)
- Support [Sparse R-CNN](https://arxiv.org/abs/2011.12450) (#4219)
- Support evaluate mAP by multiple IoUs (#4398)
- Support concatenate dataset for testing (#4452)
- Support to visualize results based on prediction quality (#4441)
- Add ONNX simplify option to Pytorch2ONNX script (#4468)
- Add hook for checking compatibility of class numbers in heads and datasets (#4508)

#### Bug Fixes

- Fix CPU inference bug of Cascade RPN (#4410)
- Fix NMS error of CornerNet when there is no prediction box (#4409)
- Fix TypeError in CornerNet inference (#4411)
- Fix bug of PAA when training with background images (#4391)
- Fix the error that the window data is not destroyed when `out_file is not None` and `show==False` (#4442)
- Fix order of NMS `score_factor` that will decrease the performance of YOLOv3 (#4473)
- Fix bug in HTC TTA when the number of detection boxes is 0 (#4516)
- Fix resize error in mask data structures (#4520)

#### Improvements

- Allow to customize classes in LVIS dataset (#4382)
- Add tutorials for building new models with existing datasets (#4396)
- Add CPU compatibility information in documentation (#4405)
- Add documentation of deprecated `ImageToTensor` for batch inference (#4408)
- Add more details in documentation for customizing dataset (#4430)
- Switch `imshow_det_bboxes` visualization backend from OpenCV to Matplotlib (#4389)
- Deprecate `ImageToTensor` in `image_demo.py` (#4400)
- Move train_cfg/test_cfg into model (#4347, #4489)
- Update docstring for `reg_decoded_bbox` option in bbox heads (#4467)
- Update dataset information in documentation (#4525)
- Release pre-trained R50 and R101 PAA detectors with multi-scale 3x training schedules (#4495)
- Add guidance for speed benchmark (#4537)

### v2.8.0 (04/01/2021)

#### Highlights

- Support new methods: [Cascade RPN](https://arxiv.org/abs/1909.06720), [TridentNet](https://arxiv.org/abs/1901.01892)

#### New Features

- Support [Cascade RPN](https://arxiv.org/abs/1909.06720) (#1900)
- Support [TridentNet](https://arxiv.org/abs/1901.01892) (#3313)

#### Bug Fixes

- Fix bug of show result in async_benchmark (#4367)
- Fix scale factor in MaskTestMixin (#4366)
- Fix but when returning indices in `multiclass_nms` (#4362)
- Fix bug of empirical attention in resnext backbone error (#4300)
- Fix bug of `img_norm_cfg` in FCOS-HRNet models with updated performance and models (#4250)
- Fix invalid checkpoint and log in Mask R-CNN models on Cityscapes dataset (#4287)
- Fix bug in distributed sampler when dataset is too small (#4257)
- Fix bug of 'PAFPN has no attribute extra_convs_on_inputs' (#4235)

#### Improvements

- Update model url from aws to aliyun (#4349)
- Update ATSS for PyTorch 1.6+ (#4359)
- Update script to install ruby in pre-commit installation (#4360)
- Delete deprecated `mmdet.ops` (#4325)
- Refactor hungarian assigner for more general usage in Sparse R-CNN (#4259)
- Handle scipy import in DETR to reduce package dependencies (#4339)
- Update documentation of usages for config options after MMCV (1.2.3) supports overriding list in config (#4326)
- Update pre-train models of faster rcnn trained on COCO subsets (#4307)
- Avoid zero or too small value for beta in Dynamic R-CNN (#4303)
- Add doccumentation for Pytorch2ONNX (#4271)
- Add deprecated warning FPN arguments (#4264)
- Support returning indices of kept bboxes when using nms (#4251)
- Update type and device requirements when creating tensors `GFLHead` (#4210)
- Update device requirements when creating tensors in `CrossEntropyLoss` (#4224)

### v2.7.0 (30/11/2020)

- Support new method: [DETR](https://arxiv.org/abs/2005.12872), [ResNest](https://arxiv.org/abs/2004.08955), Faster R-CNN DC5.
- Support YOLO, Mask R-CNN, and Cascade R-CNN models exportable to ONNX.

#### New Features

- Support [DETR](https://arxiv.org/abs/2005.12872) (#4201, #4206)
- Support to link the best checkpoint in training (#3773)
- Support to override config through options in inference.py (#4175)
- Support YOLO, Mask R-CNN, and Cascade R-CNN models exportable to ONNX (#4087, #4083)
- Support [ResNeSt](https://arxiv.org/abs/2004.08955) backbone (#2959)
- Support unclip border bbox regression (#4076)
- Add tpfp func in evaluating AP (#4069)
- Support mixed precision training of SSD detector with other backbones (#4081)
- Add Faster R-CNN DC5 models (#4043)

#### Bug Fixes

- Fix bug of `gpu_id` in distributed training mode (#4163)
- Support Albumentations with version higher than 0.5 (#4032)
- Fix num_classes bug in faster rcnn config (#4088)
- Update code in docs/2_new_data_model.md (#4041)

#### Improvements

- Ensure DCN offset to have similar type as features in VFNet (#4198)
- Add config links in README files of models (#4190)
- Add tutorials for loss conventions (#3818)
- Add solution to installation issues in 30-series GPUs (#4176)
- Update docker version in get_started.md (#4145)
- Add model statistics and polish some titles in configs README (#4140)
- Clamp neg probability in FreeAnchor (#4082)
- Speed up expanding large images (#4089)
- Fix Pytorch 1.7 incompatibility issues (#4103)
- Update trouble shooting page to resolve segmentation fault (#4055)
- Update aLRP-Loss in project page (#4078)
- Clean duplicated `reduce_mean` function (#4056)
- Refactor Q&A (#4045)

### v2.6.0 (1/11/2020)

- Support new method: [VarifocalNet](https://arxiv.org/abs/2008.13367).
- Refactored documentation with more tutorials.

#### New Features

- Support GIoU calculation in `BboxOverlaps2D`, and re-implement `giou_loss` using `bbox_overlaps` (#3936)
- Support random sampling in CPU mode (#3948)
- Support VarifocalNet (#3666, #4024)

#### Bug Fixes

- Fix SABL validating bug in Cascade R-CNN (#3913)
- Avoid division by zero in PAA head when num_pos=0 (#3938)
- Fix temporary directory bug of multi-node testing error (#4034, #4017)
- Fix `--show-dir` option in test script (#4025)
- Fix GA-RetinaNet r50 model url (#3983)
- Update code in docs and fix broken urls (#3947)

#### Improvements

- Refactor pytorch2onnx API into `mmdet.core.export` and use `generate_inputs_and_wrap_model` for pytorch2onnx (#3857, #3912)
- Update RPN upgrade scripts for v2.5.0 compatibility (#3986)
- Use mmcv `tensor2imgs` (#4010)
- Update test robustness (#4000)
- Update trouble shooting page (#3994)
- Accelerate PAA training speed (#3985)
- Support batch_size > 1 in validation (#3966)
- Use RoIAlign implemented in MMCV for inference in CPU mode (#3930)
- Documentation refactoring (#4031)

### v2.5.0 (5/10/2020)

#### Highlights

- Support new methods: [YOLACT](https://arxiv.org/abs/1904.02689), [CentripetalNet](https://arxiv.org/abs/2003.09119).
- Add more documentations for easier and more clear usage.

#### Backwards Incompatible Changes

__FP16 related methods are imported from mmcv instead of mmdet. (#3766, #3822)__
Mixed precision training utils in `mmdet.core.fp16` are moved to `mmcv.runner`, including `force_fp32`, `auto_fp16`, `wrap_fp16_model`, and `Fp16OptimizerHook`. A deprecation warning will be raised if users attempt to import those methods from `mmdet.core.fp16`, and will be finally removed in V2.10.0.

__\[0, N-1\] represents foreground classes and N indicates background classes for all models. (#3221)__
Before v2.5.0, the background label for RPN is 0, and N for other heads. Now the behavior is consistent for all models. Thus `self.background_labels` in `dense_heads` is removed and all heads use `self.num_classes` to indicate the class index of background labels.
This change has no effect on the pre-trained models in the v2.x model zoo, but will affect the training of all models with RPN heads. Two-stage detectors whose RPN head uses softmax will be affected because the order of categories is changed.

**Only call `get_subset_by_classes` when `test_mode=True` and `self.filter_empty_gt=True` (#3695)**
Function `get_subset_by_classes` in dataset is refactored and only filters out images when `test_mode=True` and `self.filter_empty_gt=True`.
In the original implementation, `get_subset_by_classes` is not related to the flag `self.filter_empty_gt` and will only be called when the classes is set during initialization no matter `test_mode` is `True` or `False`. This brings ambiguous behavior and potential bugs in many cases. After v2.5.0, if `filter_empty_gt=False`, no matter whether the classes are specified in a dataset, the dataset will use all the images in the annotations. If `filter_empty_gt=True` and `test_mode=True`, no matter whether the classes are specified, the dataset will call \`\`get_subset_by_classes\` to check the images and filter out images containing no GT boxes. Therefore, the users should be responsible for the data filtering/cleaning process for the test dataset.

#### New Features

- Test time augmentation for single stage detectors (#3844, #3638)
- Support to show the name of experiments during training (#3764)
- Add `Shear`, `Rotate`, `Translate` Augmentation (#3656, #3619, #3687)
- Add image-only transformations including `Constrast`, `Equalize`, `Color`, and `Brightness`. (#3643)
- Support [YOLACT](https://arxiv.org/abs/1904.02689) (#3456)
- Support [CentripetalNet](https://arxiv.org/abs/2003.09119) (#3390)
- Support PyTorch 1.6 in docker (#3905)

#### Bug Fixes

- Fix the bug of training ATSS when there is no ground truth boxes (#3702)
- Fix the bug of using Focal Loss when there is `num_pos` is 0 (#3702)
- Fix the label index mapping in dataset browser (#3708)
- Fix Mask R-CNN training stuck problem when their is no positive rois (#3713)
- Fix the bug of `self.rpn_head.test_cfg` in `RPNTestMixin` by using `self.rpn_head` in rpn head (#3808)
- Fix deprecated `Conv2d` from mmcv.ops (#3791)
- Fix device bug in RepPoints (#3836)
- Fix SABL validating bug (#3849)
- Use `https://download.openmmlab.com/mmcv/dist/index.html` for installing MMCV (#3840)
- Fix nonzero in NMS for PyTorch 1.6.0 (#3867)
- Fix the API change bug of PAA (#3883)
- Fix typo in bbox_flip (#3886)
- Fix cv2 import error of ligGL.so.1 in Dockerfile (#3891)

#### Improvements

- Change to use `mmcv.utils.collect_env` for collecting environment information to avoid duplicate codes (#3779)
- Update checkpoint file names to v2.0 models in documentation (#3795)
- Update tutorials for changing runtime settings (#3778), modifying loss (#3777)
- Improve the function of `simple_test_bboxes` in SABL (#3853)
- Convert mask to bool before using it as img's index for robustness and speedup (#3870)
- Improve documentation of modules and dataset customization (#3821)

### v2.4.0 (5/9/2020)

__Highlights__

- Fix lots of issues/bugs and reorganize the trouble shooting page
- Support new methods [SABL](https://arxiv.org/abs/1912.04260), [YOLOv3](https://arxiv.org/abs/1804.02767), and [PAA Assign](https://arxiv.org/abs/2007.08103)
- Support Batch Inference
- Start to publish `mmdet` package to PyPI since v2.3.0
- Switch model zoo to download.openmmlab.com

__Backwards Incompatible Changes__

- Support Batch Inference (#3564, #3686, #3705): Since v2.4.0, MMDetection could inference model with multiple images in a single GPU.
  This change influences all the test APIs in MMDetection and downstream codebases. To help the users migrate their code, we use `replace_ImageToTensor` (#3686) to convert legacy test data pipelines during dataset initialization.
- Support RandomFlip with horizontal/vertical/diagonal direction (#3608): Since v2.4.0, MMDetection supports horizontal/vertical/diagonal flip in the data augmentation. This influences bounding box, mask, and image transformations in data augmentation process and the process that will map those data back to the original format.
- Migrate to use `mmlvis` and `mmpycocotools` for COCO and LVIS dataset (#3727). The APIs are fully compatible with the original `lvis` and `pycocotools`. Users need to uninstall the existing pycocotools and lvis packages in their environment first and install `mmlvis` & `mmpycocotools`.

__Bug Fixes__

- Fix default mean/std for onnx (#3491)
- Fix coco evaluation and add metric items (#3497)
- Fix typo for install.md (#3516)
- Fix atss when sampler per gpu is 1 (#3528)
- Fix import of fuse_conv_bn (#3529)
- Fix bug of gaussian_target, update unittest of heatmap (#3543)
- Fixed VOC2012 evaluate (#3553)
- Fix scale factor bug of rescale (#3566)
- Fix with_xxx_attributes in base detector (#3567)
- Fix boxes scaling when number is 0 (#3575)
- Fix rfp check when neck config is a list (#3591)
- Fix import of fuse conv bn in benchmark.py (#3606)
- Fix webcam demo (#3634)
- Fix typo and itemize issues in tutorial (#3658)
- Fix error in distributed training when some levels of FPN are not assigned with bounding boxes (#3670)
- Fix the width and height orders of stride in valid flag generation (#3685)
- Fix weight initialization bug in Res2Net DCN (#3714)
- Fix bug in OHEMSampler (#3677)

__New Features__

- Support Cutout augmentation (#3521)
- Support evaluation on multiple datasets through ConcatDataset (#3522)
- Support [PAA assign](https://arxiv.org/abs/2007.08103) #(3547)
- Support eval metric with pickle results (#3607)
- Support [YOLOv3](https://arxiv.org/abs/1804.02767) (#3083)
- Support [SABL](https://arxiv.org/abs/1912.04260) (#3603)
- Support to publish to Pypi in github-action (#3510)
- Support custom imports (#3641)

__Improvements__

- Refactor common issues in documentation (#3530)
- Add pytorch 1.6 to CI config (#3532)
- Add config to runner meta (#3534)
- Add eval-option flag for testing (#3537)
- Add init_eval to evaluation hook (#3550)
- Add include_bkg in ClassBalancedDataset (#3577)
- Using config's loading in inference_detector (#3611)
- Add ATSS ResNet-101 models in model zoo (#3639)
- Update urls to download.openmmlab.com (#3665)
- Support non-mask training for CocoDataset (#3711)

### v2.3.0 (5/8/2020)

__Highlights__

- The CUDA/C++ operators have been moved to `mmcv.ops`. For backward compatibility `mmdet.ops` is kept as warppers of `mmcv.ops`.
- Support new methods [CornerNet](https://arxiv.org/abs/1808.01244), [DIOU](https://arxiv.org/abs/1911.08287)/[CIOU](https://arxiv.org/abs/2005.03572) loss, and new dataset: [LVIS V1](https://arxiv.org/abs/1908.03195)
- Provide more detailed colab training tutorials and more complete documentation.
- Support to convert RetinaNet from Pytorch to ONNX.

__Bug Fixes__

- Fix the model initialization bug of DetectoRS (#3187)
- Fix the bug of module names in NASFCOSHead (#3205)
- Fix the filename bug in publish_model.py (#3237)
- Fix the dimensionality bug when `inside_flags.any()` is `False` in dense heads (#3242)
- Fix the bug of forgetting to pass flip directions in `MultiScaleFlipAug` (#3262)
- Fixed the bug caused by default value of `stem_channels` (#3333)
- Fix the bug of model checkpoint loading for CPU inference (#3318, #3316)
- Fix topk bug when box number is smaller than the expected topk number in ATSSAssigner (#3361)
- Fix the gt priority bug in center_region_assigner.py (#3208)
- Fix NaN issue of iou calculation in iou_loss.py (#3394)
- Fix the bug that `iou_thrs` is not actually used during evaluation in coco.py (#3407)
- Fix test-time augmentation of RepPoints (#3435)
- Fix runtimeError caused by incontiguous tensor in Res2Net+DCN (#3412)

__New Features__

- Support [CornerNet](https://arxiv.org/abs/1808.01244) (#3036)
- Support [DIOU](https://arxiv.org/abs/1911.08287)/[CIOU](https://arxiv.org/abs/2005.03572) loss (#3151)
- Support [LVIS V1](https://arxiv.org/abs/1908.03195) dataset (#)
- Support customized hooks in training (#3395)
- Support fp16 training of generalized focal loss (#3410)
- Support to convert RetinaNet from Pytorch to ONNX (#3075)

__Improvements__

- Support to process ignore boxes in ATSS assigner (#3082)
- Allow to crop images without ground truth in `RandomCrop` (#3153)
- Enable the the `Accuracy` module to set threshold (#3155)
- Refactoring unit tests (#3206)
- Unify the training settings of `to_float32` and `norm_cfg` in RegNets configs (#3210)
- Add colab training tutorials for beginners (#3213, #3273)
- Move CUDA/C++ operators into `mmcv.ops` and keep `mmdet.ops` as warppers for backward compatibility (#3232)(#3457)
- Update installation scripts in documentation (#3290) and dockerfile (#3320)
- Support to set image resize backend (#3392)
- Remove git hash in version file (#3466)
- Check mmcv version to force version compatibility (#3460)

### v2.2.0 (1/7/2020)

__Highlights__

- Support new methods: [DetectoRS](https://arxiv.org/abs/2006.02334), [PointRend](https://arxiv.org/abs/1912.08193), [Generalized Focal Loss](https://arxiv.org/abs/2006.04388), [Dynamic R-CNN](https://arxiv.org/abs/2004.06002)

__Bug Fixes__

- Fix FreeAnchor when no gt in image (#3176)
- Clean up deprecated usage of `register_module()` (#3092, #3161)
- Fix pretrain bug in NAS FCOS (#3145)
- Fix `num_classes` in SSD (#3142)
- Fix FCOS warmup (#3119)
- Fix `rstrip` in `tools/publish_model.py`
- Fix `flip_ratio` default value in RandomFLip pipeline (#3106)
- Fix cityscapes eval with ms_rcnn (#3112)
- Fix RPN softmax (#3056)
- Fix filename of LVIS@v0.5 (#2998)
- Fix nan loss by filtering out-of-frame gt_bboxes in COCO (#2999)
- Fix bug in FSAF (#3018)
- Add FocalLoss `num_classes` check (#2964)
- Fix PISA Loss when there are no gts (#2992)
- Avoid nan in `iou_calculator` (#2975)
- Prevent possible bugs in loading and transforms caused by shallow copy (#2967)

__New Features__

- Add DetectoRS (#3064)
- Support Generalize Focal Loss (#3097)
- Support PointRend (#2752)
- Support Dynamic R-CNN (#3040)
- Add DeepFashion dataset (#2968)
- Implement FCOS training tricks (#2935)
- Use BaseDenseHead as base class for anchor-base heads (#2963)
- Add `with_cp` for BasicBlock (#2891)
- Add `stem_channels` argument for ResNet (#2954)

__Improvements__

- Add anchor free base head (#2867)
- Migrate to github action (#3137)
- Add docstring for datasets, pipelines, core modules and methods (#3130, #3125, #3120)
- Add VOC benchmark (#3060)
- Add `concat` mode in GRoI (#3098)
- Remove cmd arg `autorescale-lr` (#3080)
- Use `len(data['img_metas'])` to indicate `num_samples` (#3073, #3053)
- Switch to EpochBasedRunner (#2976)

### v2.1.0 (8/6/2020)

__Highlights__

- Support new backbones: [RegNetX](https://arxiv.org/abs/2003.13678), [Res2Net](https://arxiv.org/abs/1904.01169)
- Support new methods: [NASFCOS](https://arxiv.org/abs/1906.04423), [PISA](https://arxiv.org/abs/1904.04821), [GRoIE](https://arxiv.org/abs/2004.13665)
- Support new dataset: [LVIS](https://arxiv.org/abs/1908.03195)

__Bug Fixes__

- Change the CLI argument `--validate` to `--no-validate` to enable validation after training epochs by default. (#2651)
- Add missing cython to docker file (#2713)
- Fix bug in nms cpu implementation (#2754)
- Fix bug when showing mask results (#2763)
- Fix gcc requirement (#2806)
- Fix bug in async test (#2820)
- Fix mask encoding-decoding bugs in test API (#2824)
- Fix bug in test time augmentation (#2858, #2921, #2944)
- Fix a typo in comment of apis/train (#2877)
- Fix the bug of returning None when no gt bboxes are in the original image in `RandomCrop`. Fix the bug that misses to handle `gt_bboxes_ignore`, `gt_label_ignore`, and `gt_masks_ignore` in `RandomCrop`, `MinIoURandomCrop` and `Expand` modules. (#2810)
- Fix bug of `base_channels` of regnet (#2917)
- Fix the bug of logger when loading pre-trained weights in base detector (#2936)

__New Features__

- Add IoU models (#2666)
- Add colab demo for inference
- Support class agnostic nms (#2553)
- Add benchmark gathering scripts for development only (#2676)
- Add mmdet-based project links (#2736, #2767, #2895)
- Add config dump in training (#2779)
- Add ClassBalancedDataset (#2721)
- Add res2net backbone (#2237)
- Support RegNetX models (#2710)
- Use `mmcv.FileClient` to support different storage backends (#2712)
- Add ClassBalancedDataset (#2721)
- Code Release: Prime Sample Attention in Object Detection (CVPR 2020) (#2626)
- Implement NASFCOS (#2682)
- Add class weight in CrossEntropyLoss (#2797)
- Support LVIS dataset (#2088)
- Support GRoIE (#2584)

__Improvements__

- Allow different x and y strides in anchor heads. (#2629)
- Make FSAF loss more robust to no gt (#2680)
- Compute pure inference time instead (#2657) and update inference speed (#2730)
- Avoided the possibility that a patch with 0 area is cropped. (#2704)
- Add warnings when deprecated `imgs_per_gpu` is used. (#2700)
- Add a mask rcnn example for config (#2645)
- Update model zoo (#2762, #2866, #2876, #2879, #2831)
- Add `ori_filename` to img_metas and use it in test show-dir (#2612)
- Use `img_fields` to handle multiple images during image transform (#2800)
- Add upsample_cfg support in FPN (#2787)
- Add `['img']` as default `img_fields` for back compatibility (#2809)
- Rename the pretrained model from `open-mmlab://resnet50_caffe` and `open-mmlab://resnet50_caffe_bgr` to `open-mmlab://detectron/resnet50_caffe` and `open-mmlab://detectron2/resnet50_caffe`. (#2832)
- Added sleep(2) in test.py to reduce hanging problem (#2847)
- Support `c10::half` in CARAFE (#2890)
- Improve documentations (#2918, #2714)
- Use optimizer constructor in mmcv and clean the original implementation in `mmdet.core.optimizer` (#2947)

### v2.0.0 (6/5/2020)

In this release, we made lots of major refactoring and modifications.

1. __Faster speed__. We optimize the training and inference speed for common models, achieving up to 30% speedup for training and 25% for inference. Please refer to [model zoo](model_zoo.md#comparison-with-detectron2) for details.

2. __Higher performance__. We change some default hyperparameters with no additional cost, which leads to a gain of performance for most models. Please refer to [compatibility](compatibility.md#training-hyperparameters) for details.

3. __More documentation and tutorials__. We add a bunch of documentation and tutorials to help users get started more smoothly. Read it [here](https://mmdetection.readthedocs.io/en/latest/).

4. __Support PyTorch 1.5__. The support for 1.1 and 1.2 is dropped, and we switch to some new APIs.

5. __Better configuration system__. Inheritance is supported to reduce the redundancy of configs.

6. __Better modular design__. Towards the goal of simplicity and flexibility, we simplify some encapsulation while add more other configurable modules like BBoxCoder, IoUCalculator, OptimizerConstructor, RoIHead. Target computation is also included in heads and the call hierarchy is simpler.

7. Support new methods: [FSAF](https://arxiv.org/abs/1903.00621) and PAFPN (part of [PAFPN](https://arxiv.org/abs/1803.01534)).

__Breaking Changes__
Models training with MMDetection 1.x are not fully compatible with 2.0, please refer to the [compatibility doc](compatibility.md) for the details and how to migrate to the new version.

__Improvements__

- Unify cuda and cpp API for custom ops. (#2277)
- New config files with inheritance. (#2216)
- Encapsulate the second stage into RoI heads. (#1999)
- Refactor GCNet/EmpericalAttention into plugins. (#2345)
- Set low quality match as an option in IoU-based bbox assigners. (#2375)
- Change the codebase's coordinate system. (#2380)
- Refactor the category order in heads. 0 means the first positive class instead of background now. (#2374)
- Add bbox sampler and assigner registry. (#2419)
- Speed up the inference of RPN. (#2420)
- Add `train_cfg` and `test_cfg` as class members in all anchor heads. (#2422)
- Merge target computation methods into heads. (#2429)
- Add bbox coder to support different bbox encoding and losses. (#2480)
- Unify the API for regression loss. (#2156)
- Refactor Anchor Generator. (#2474)
- Make `lr` an optional argument for optimizers. (#2509)
- Migrate to modules and methods in MMCV. (#2502, #2511, #2569, #2572)
- Support PyTorch 1.5. (#2524)
- Drop the support for Python 3.5 and use F-string in the codebase. (#2531)

__Bug Fixes__

- Fix the scale factors for resized images without keep the aspect ratio. (#2039)
- Check if max_num > 0 before slicing in NMS. (#2486)
- Fix Deformable RoIPool when there is no instance. (#2490)
- Fix the default value of assigned labels. (#2536)
- Fix the evaluation of Cityscapes. (#2578)

__New Features__

- Add deep_stem and avg_down option to ResNet, i.e., support ResNetV1d. (#2252)
- Add L1 loss. (#2376)
- Support both polygon and bitmap for instance masks. (#2353, #2540)
- Support CPU mode for inference. (#2385)
- Add optimizer constructor for complicated configuration of optimizers. (#2397, #2488)
- Implement PAFPN. (#2392)
- Support empty tensor input for some modules. (#2280)
- Support for custom dataset classes without overriding it. (#2408, #2443)
- Support to train subsets of coco dataset. (#2340)
- Add iou_calculator to potentially support more IoU calculation methods. (2405)
- Support class wise mean AP (was removed in the last version). (#2459)
- Add option to save the testing result images. (#2414)
- Support MomentumUpdaterHook. (#2571)
- Add a demo to inference a single image. (#2605)

### v1.1.0 (24/2/2020)

__Highlights__

- Dataset evaluation is rewritten with a unified api, which is used by both evaluation hooks and test scripts.
- Support new methods: [CARAFE](https://arxiv.org/abs/1905.02188).

__Breaking Changes__

- The new MMDDP inherits from the official DDP, thus the `__init__` api is changed to be the same as official DDP.
- The `mask_head` field in HTC config files is modified.
- The evaluation and testing script is updated.
- In all transforms, instance masks are stored as a numpy array shaped (n, h, w) instead of a list of (h, w) arrays, where n is the number of instances.

__Bug Fixes__

- Fix IOU assigners when ignore_iof_thr > 0 and there is no pred boxes. (#2135)
- Fix mAP evaluation when there are no ignored boxes. (#2116)
- Fix the empty RoI input for Deformable RoI Pooling. (#2099)
- Fix the dataset settings for multiple workflows. (#2103)
- Fix the warning related to `torch.uint8` in PyTorch 1.4. (#2105)
- Fix the inference demo on devices other than gpu:0. (#2098)
- Fix Dockerfile. (#2097)
- Fix the bug that `pad_val` is unused in Pad transform. (#2093)
- Fix the albumentation transform when there is no ground truth bbox. (#2032)

__Improvements__

- Use torch instead of numpy for random sampling. (#2094)
- Migrate to the new MMDDP implementation in MMCV v0.3. (#2090)
- Add meta information in logs. (#2086)
- Rewrite Soft NMS with pytorch extension and remove cython as a dependency. (#2056)
- Rewrite dataset evaluation. (#2042, #2087, #2114, #2128)
- Use numpy array for masks in transforms. (#2030)

__New Features__

- Implement "CARAFE: Content-Aware ReAssembly of FEatures". (#1583)
- Add `worker_init_fn()` in data_loader when seed is set. (#2066, #2111)
- Add logging utils. (#2035)

### v1.0.0 (30/1/2020)

This release mainly improves the code quality and add more docstrings.

__Highlights__

- Documentation is online now: <https://mmdetection.readthedocs.io>.
- Support new models: [ATSS](https://arxiv.org/abs/1912.02424).
- DCN is now available with the api `build_conv_layer` and `ConvModule` like the normal conv layer.
- A tool to collect environment information is available for trouble shooting.

__Bug Fixes__

- Fix the incompatibility of the latest numpy and pycocotools. (#2024)
- Fix the case when distributed package is unavailable, e.g., on Windows. (#1985)
- Fix the dimension issue for `refine_bboxes()`. (#1962)
- Fix the typo when `seg_prefix` is a list. (#1906)
- Add segmentation map cropping to RandomCrop. (#1880)
- Fix the return value of `ga_shape_target_single()`. (#1853)
- Fix the loaded shape of empty proposals. (#1819)
- Fix the mask data type when using albumentation. (#1818)

__Improvements__

- Enhance AssignResult and SamplingResult. (#1995)
- Add ability to overwrite existing module in Registry. (#1982)
- Reorganize requirements and make albumentations and imagecorruptions optional. (#1969)
- Check NaN in `SSDHead`. (#1935)
- Encapsulate the DCN in ResNe(X)t into a ConvModule & Conv_layers. (#1894)
- Refactoring for mAP evaluation and support multiprocessing and logging. (#1889)
- Init the root logger before constructing Runner to log more information. (#1865)
- Split `SegResizeFlipPadRescale` into different existing transforms. (#1852)
- Move `init_dist()` to MMCV. (#1851)
- Documentation and docstring improvements. (#1971, #1938, #1869, #1838)
- Fix the color of the same class for mask visualization. (#1834)
- Remove the option `keep_all_stages` in HTC and Cascade R-CNN. (#1806)

__New Features__

- Add two test-time options `crop_mask` and `rle_mask_encode` for mask heads. (#2013)
- Support loading grayscale images as single channel. (#1975)
- Implement "Bridging the Gap Between Anchor-based and Anchor-free Detection via Adaptive Training Sample Selection". (#1872)
- Add sphinx generated docs. (#1859, #1864)
- Add GN support for flops computation. (#1850)
- Collect env info for trouble shooting. (#1812)

### v1.0rc1 (13/12/2019)

The RC1 release mainly focuses on improving the user experience, and fixing bugs.

__Highlights__

- Support new models: [FoveaBox](https://arxiv.org/abs/1904.03797), [RepPoints](https://arxiv.org/abs/1904.11490) and [FreeAnchor](https://arxiv.org/abs/1909.02466).
- Add a Dockerfile.
- Add a jupyter notebook demo and a webcam demo.
- Setup the code style and CI.
- Add lots of docstrings and unit tests.
- Fix lots of bugs.

__Breaking Changes__

- There was a bug for computing COCO-style mAP w.r.t different scales (AP_s, AP_m, AP_l), introduced by #621. (#1679)

__Bug Fixes__

- Fix a sampling interval bug in Libra R-CNN. (#1800)
- Fix the learning rate in SSD300 WIDER FACE. (#1781)
- Fix the scaling issue when `keep_ratio=False`. (#1730)
- Fix typos. (#1721, #1492, #1242, #1108, #1107)
- Fix the shuffle argument in `build_dataloader`. (#1693)
- Clip the proposal when computing mask targets. (#1688)
- Fix the "index out of range" bug for samplers in some corner cases. (#1610, #1404)
- Fix the NMS issue on devices other than GPU:0. (#1603)
- Fix SSD Head and GHM Loss on CPU. (#1578)
- Fix the OOM error when there are too many gt bboxes. (#1575)
- Fix the wrong keyword argument `nms_cfg` in HTC. (#1573)
- Process masks and semantic segmentation in Expand and MinIoUCrop transforms. (#1550, #1361)
- Fix a scale bug in the Non Local op. (#1528)
- Fix a bug in transforms when `gt_bboxes_ignore` is None. (#1498)
- Fix a bug when `img_prefix` is None. (#1497)
- Pass the device argument to `grid_anchors` and `valid_flags`. (#1478)
- Fix the data pipeline for test_robustness. (#1476)
- Fix the argument type of deformable pooling. (#1390)
- Fix the coco_eval when there are only two classes. (#1376)
- Fix a bug in Modulated DeformableConv when deformable_group>1. (#1359)
- Fix the mask cropping in RandomCrop. (#1333)
- Fix zero outputs in DeformConv when not running on cuda:0. (#1326)
- Fix the type issue in Expand. (#1288)
- Fix the inference API. (#1255)
- Fix the inplace operation in Expand. (#1249)
- Fix the from-scratch training config. (#1196)
- Fix inplace add in RoIExtractor which cause an error in PyTorch 1.2. (#1160)
- Fix FCOS when input images has no positive sample. (#1136)
- Fix recursive imports. (#1099)

__Improvements__

- Print the config file and mmdet version in the log. (#1721)
- Lint the code before compiling in travis CI. (#1715)
- Add a probability argument for the `Expand` transform. (#1651)
- Update the PyTorch and CUDA version in the docker file. (#1615)
- Raise a warning when specifying `--validate` in non-distributed training. (#1624, #1651)
- Beautify the mAP printing. (#1614)
- Add pre-commit hook. (#1536)
- Add the argument `in_channels` to backbones. (#1475)
- Add lots of docstrings and unit tests, thanks to [@Erotemic](https://github.com/Erotemic). (#1603, #1517, #1506, #1505, #1491, #1479, #1477, #1475, #1474)
- Add support for multi-node distributed test when there is no shared storage. (#1399)
- Optimize Dockerfile to reduce the image size. (#1306)
- Update new results of HRNet. (#1284, #1182)
- Add an argument `no_norm_on_lateral` in FPN. (#1240)
- Test the compiling in CI. (#1235)
- Move docs to a separate folder. (#1233)
- Add a jupyter notebook demo. (#1158)
- Support different type of dataset for training. (#1133)
- Use int64_t instead of long in cuda kernels. (#1131)
- Support unsquare RoIs for bbox and mask heads. (#1128)
- Manually add type promotion to make compatible to PyTorch 1.2. (#1114)
- Allowing validation dataset for computing validation loss. (#1093)
- Use `.scalar_type()` instead of `.type()` to suppress some warnings. (#1070)

__New Features__

- Add an option `--with_ap` to compute the AP for each class. (#1549)
- Implement "FreeAnchor: Learning to Match Anchors for Visual Object Detection". (#1391)
- Support [Albumentations](https://github.com/albumentations-team/albumentations) for augmentations in the data pipeline. (#1354)
- Implement "FoveaBox: Beyond Anchor-based Object Detector". (#1339)
- Support horizontal and vertical flipping. (#1273, #1115)
- Implement "RepPoints: Point Set Representation for Object Detection". (#1265)
- Add test-time augmentation to HTC and Cascade R-CNN. (#1251)
- Add a COCO result analysis tool. (#1228)
- Add Dockerfile. (#1168)
- Add a webcam demo. (#1155, #1150)
- Add FLOPs counter. (#1127)
- Allow arbitrary layer order for ConvModule. (#1078)

### v1.0rc0 (27/07/2019)

- Implement lots of new methods and components (Mixed Precision Training, HTC, Libra R-CNN, Guided Anchoring, Empirical Attention, Mask Scoring R-CNN, Grid R-CNN (Plus), GHM, GCNet, FCOS, HRNet, Weight Standardization, etc.). Thank all collaborators!
- Support two additional datasets: WIDER FACE and Cityscapes.
- Refactoring for loss APIs and make it more flexible to adopt different losses and related hyper-parameters.
- Speed up multi-gpu testing.
- Integrate all compiling and installing in a single script.

### v0.6.0 (14/04/2019)

- Up to 30% speedup compared to the model zoo.
- Support both PyTorch stable and nightly version.
- Replace NMS and SigmoidFocalLoss with Pytorch CUDA extensions.

### v0.6rc0(06/02/2019)

- Migrate to PyTorch 1.0.

### v0.5.7 (06/02/2019)

- Add support for Deformable ConvNet v2. (Many thanks to the authors and [@chengdazhi](https://github.com/chengdazhi))
- This is the last release based on PyTorch 0.4.1.

### v0.5.6 (17/01/2019)

- Add support for Group Normalization.
- Unify RPNHead and single stage heads (RetinaHead, SSDHead) with AnchorHead.

### v0.5.5 (22/12/2018)

- Add SSD for COCO and PASCAL VOC.
- Add ResNeXt backbones and detection models.
- Refactoring for Samplers/Assigners and add OHEM.
- Add VOC dataset and evaluation scripts.

### v0.5.4 (27/11/2018)

- Add SingleStageDetector and RetinaNet.

### v0.5.3 (26/11/2018)

- Add Cascade R-CNN and Cascade Mask R-CNN.
- Add support for Soft-NMS in config files.

### v0.5.2 (21/10/2018)

- Add support for custom datasets.
- Add a script to convert PASCAL VOC annotations to the expected format.

### v0.5.1 (20/10/2018)

- Add BBoxAssigner and BBoxSampler, the `train_cfg` field in config files are restructured.
- `ConvFCRoIHead` / `SharedFCRoIHead` are renamed to `ConvFCBBoxHead` / `SharedFCBBoxHead` for consistency.

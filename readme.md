# FunSR

Official Implement of the Paper "Continuous Remote Sensing Image Super-Resolution based on Context Interaction in Implicit Function Space"

## Some Information

[Project Page](https://kyanchen.github.io/FunSR/) $\cdot$ [PDF Download](https://arxiv.org/abs/2302.08046)

🚀️🚀️🚀️ The repository will be orginized later.

## How to train a SR model

### Fixed scale SR model (*e.g.*, TransEnet, SRCNN, LGCNet, FSRCNN, DCM, VDSR)

Please run **"train_cnn_sr.py"**. Config is the file of **"configs/train_CNN.yaml**.

### Continuous scale SR model (*e.g.*, LIIF, MetaSR, ALIIF)

Please run **"train_liif_metasr_aliff.py"**. Config are in the folder of **"configs/baselines/"**.

### Continuous scale SR model (*e.g.*, DIINN, ArbRCAN, SADN, OverNet)

Please run **"train_diinn_arbrcan_sadn_overnet.py"**. Config is the file of **"configs/baselines/train_1x-5x_INR_diinn_arbrcan_sadn_overnet.yaml"**.

### **Our FunSR model**

Please run **"train_inr_funsr.py"** for a single GPU training or DP multi GPUs training. Config is the file of **"configs/train_1x-5x_INR_funsr.yaml"**.

👍👍👍 We also provide a DDP multi GPUs training method. Please refer to **"scripts/train_multi_gpus_with_ddp.py"** and **"train_inr_funsr_ddp.py"** for more details.

## How to eval a SR model

### Interpolation SR model (*e.g.*, Bilinear, Bicubic)

Please run **"test_interpolate_sr.py"**. Config is the file of **"configs/test_interpolate.yaml**.

### Fixed scale SR model (*e.g.*, TransEnet, SRCNN, LGCNet, FSRCNN, DCM, VDSR)

Please run **"test_cnn_sr.py"**. Config is the file of **"configs/test_CNN.yaml**.

### Continuous scale SR model (*e.g.*, LIIF, MetaSR, ALIIF)

Please run **"test_inr_liif_metasr_aliif.py"**. Config is the file of **"configs/baselines/test_INR_liif_metasr_aliif.yaml"**.

### Continuous scale SR model (*e.g.*, DIINN, ArbRCAN, SADN, OverNet, **FunSR**)

Please run **"test_inr_diinn_arbrcan_sadnarc_funsr_overnet.py"**. Config is the file of **"configs/test_INR_diinn_arbrcan_funsr_overnet.yaml"**.

👍👍👍 We also provide a script to evaluate multi upscale factors in a batch. Please refer to **"scripts/test_script.py"** for more details.

## Result visualization

We provide some visualization tools to show the reconstruction results in our paper. Please refer to the folder **"tools/paper_vis_tools"** for more details.




```
@misc{chen2023continuous,
      title={Continuous Remote Sensing Image Super-Resolution based on Context Interaction in Implicit Function Space}, 
      author={Keyan Chen and Wenyuan Li and Sen Lei and Jianqi Chen and Xiaolong Jiang and Zhengxia Zou and Zhenwei Shi},
      year={2023},
      eprint={2302.08046},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

If you have any questions, please feel free to reach me.


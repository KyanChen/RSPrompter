import argparse
import os
import sys
sys.path.insert(0, sys.path[0]+'/..')
from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner
from mmpl.engine.runner import PLRunner
import os.path as osp
from mmpl.registry import RUNNERS
from mmpl.utils import register_all_modules
register_all_modules()


def parse_args():
    parser = argparse.ArgumentParser(description='Train a pl model')
    parser.add_argument('--config', default='configs/motion/motiongpt_huggpt_kit_config.py', help='train config file path')
    parser.add_argument('--status', default='test', help='fit or test', choices=['fit', 'test', 'predict', 'validate'])
    # parser.add_argument('--ckpt-path', default='results/exp/MotionGPT/tc6u6115/checkpoints/epoch=55-step=88088.ckpt', help='checkpoint path')
    parser.add_argument('--ckpt-path', default='pretrain/motiongpt/LM_GPT_E20230510_0.pth',
                        help='checkpoint path')
    parser.add_argument('--work-dir', default=None, help='the dir to save logs and mmpl')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.work_dir is not None:
        cfg.trainer_cfg['default_root_dir'] = args.work_dir
    elif cfg.trainer_cfg.get('default_root_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.trainer_cfg['default_root_dir'] = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])

    if 'runner_type' not in cfg:
        runner = PLRunner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)
    runner.run(args.status, ckpt_path=args.ckpt_path)


if __name__ == '__main__':
    main()


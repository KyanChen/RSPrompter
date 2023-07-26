import copy
import logging
import os
import os.path as osp
import pickle
import platform
import time
import warnings
from collections import OrderedDict
from functools import partial
from typing import Callable, Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
from lightning.pytorch.loggers import Logger
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.optim import Optimizer
from torch.utils.data import DataLoader

import mmengine
from mmengine.config import Config, ConfigDict
from mmengine.dataset import worker_init_fn
from mmengine.device import get_device
from mmengine.dist import (broadcast, get_dist_info, get_rank, init_dist,
                           is_distributed, master_only)
from mmengine.evaluator import Evaluator
from mmengine.fileio import FileClient, join_path
from mmengine.hooks import Hook
from mmengine.logging import MessageHub, MMLogger, print_log
from mmengine.model import (MMDistributedDataParallel, convert_sync_batchnorm,
                            is_model_wrapper, revert_sync_batchnorm)
from mmengine.optim import (OptimWrapper, OptimWrapperDict, _ParamScheduler,
                            build_optim_wrapper)
from mmengine.registry import (DATA_SAMPLERS, DATASETS, EVALUATOR, FUNCTIONS,
                               HOOKS, LOG_PROCESSORS, LOOPS, MODEL_WRAPPERS,
                               OPTIM_WRAPPERS, PARAM_SCHEDULERS,
                               RUNNERS, VISUALIZERS, DefaultScope)
from mmengine.utils import digit_version, get_git_hash, is_seq_of
from mmengine.utils.dl_utils import (TORCH_VERSION, collect_env,
                                     set_multi_processing)
from mmengine.visualization import Visualizer
from mmengine.runner.base_loop import BaseLoop
from mmengine.runner.checkpoint import (_load_checkpoint, _load_checkpoint_to_model,
                         find_latest_checkpoint, get_state_dict,
                         save_checkpoint, weights_to_cpu)
from mmengine.runner.log_processor import LogProcessor
from mmengine.runner.loops import EpochBasedTrainLoop, IterBasedTrainLoop, TestLoop, ValLoop
from mmengine.runner.priority import Priority, get_priority
from mmengine.runner.utils import set_random_seed

ConfigType = Union[Dict, Config, ConfigDict]
ParamSchedulerType = Union[List[_ParamScheduler], Dict[str, List[_ParamScheduler]]]
OptimWrapperType = Union[OptimWrapper, OptimWrapperDict]

from mmpl.registry import MODELS, LOGGERS
import lightning.pytorch as pl
from mmpl.models import build_pler


@RUNNERS.register_module()
class PLRunner:
    def __init__(
        self,
        trainer_cfg: Dict,
        model_cfg: Union[pl.LightningModule, Dict],
        datamodule_cfg: Optional[Dict] = None,
        cfg: Optional[ConfigType] = None
    ):
        self.trainer_cfg = copy.deepcopy(trainer_cfg)
        self.model_cfg = copy.deepcopy(model_cfg)
        self.datamodule_cfg = copy.deepcopy(datamodule_cfg)
        mmengine.mkdir_or_exist(trainer_cfg['default_root_dir'])

        timestamp = torch.tensor(time.time(), dtype=torch.float64)
        # broadcast timestamp from 0 process to other processes
        broadcast(timestamp)
        self.timestamp = time.strftime('%Y%m%d_%H%M%S',
                                       time.localtime(timestamp.item()))

        if cfg is not None:
            if isinstance(cfg, Config):
                self.cfg = copy.deepcopy(cfg)
            elif isinstance(cfg, dict):
                self.cfg = Config(cfg)
        else:
            self.cfg = Config(dict())

        compiled_model = trainer_cfg.pop('compiled_model', False)

        # build logger
        loggers = self.build_logger(
            trainer_cfg.get('logger', False),
            trainer_cfg.get('default_root_dir', f'{self.timestamp}')
        )
        trainer_cfg['logger'] = loggers

        # build visualizer used for writing log or visualizing all kinds of data
        self.visualizer = self.build_visualizer(
            self.cfg.get('visualizer', None),
            trainer_cfg.get('default_root_dir', f'{self.timestamp}')
        )
        if self.cfg:
            self.visualizer.add_config(self.cfg)

        # build callbacks
        callbacks = self.build_hooks(
            trainer_cfg.get('callbacks', None),
        )
        trainer_cfg['callbacks'] = callbacks

        # build strategy
        strategy = self.build_strategy(
            trainer_cfg.get('strategy', 'auto'),
        )
        trainer_cfg['strategy'] = strategy

        self.trainer = pl.Trainer(**trainer_cfg)
        model_cfg.update({'config_cfg': ConfigDict(copy.deepcopy(cfg).to_dict())})
        model = self.build_model(model_cfg)
        if cfg.get('load_from', None) is not None:
            self.load_checkpoint(model, cfg['load_from'])
        if compiled_model:
            # default, reduce-overhead, and max-autotune.
            self.model = torch.compile(model)
        else:
            self.model = model

        # dump `cfg` to `work_dir`
        self.dump_config()
        # # Collect and log environment information.
        # self._log_env(env_cfg)
        # log hooks information
        # self.logger.info(f'Hooks will be executed in the following '
        #                  f'order:\n{self.get_hooks_info()}')

    def build_visualizer(
            self,
            visualizer: Optional[Union[Visualizer,
                                       Dict]] = None,
            default_root_dir = 'tmp'
    ) -> Visualizer:
        """Build a global asscessable Visualizer.

        Args:
            visualizer (Visualizer or dict, optional): A Visualizer object
                or a dict to build Visualizer object. If ``visualizer`` is a
                Visualizer object, just returns itself. If not specified,
                default config will be used to build Visualizer object.
                Defaults to None.

        Returns:
            Visualizer: A Visualizer object build from ``visualizer``.
        """
        if visualizer is None:
            visualizer = dict(
                name=os.path.basename(default_root_dir),
                vis_backends=[dict(type='LocalVisBackend')],
                save_dir=default_root_dir+'/visualizer'
            )
            return Visualizer.get_instance(**visualizer)

        if isinstance(visualizer, Visualizer):
            return visualizer

        if isinstance(visualizer, dict):
            # ensure visualizer containing name key
            visualizer.setdefault('name', os.path.basename(default_root_dir))
            visualizer.setdefault('save_dir', default_root_dir+'/visualizer')
            return VISUALIZERS.build(visualizer)
        else:
            raise TypeError(
                'visualizer should be Visualizer object, a dict or None, '
                f'but got {visualizer}')

    def build_hooks(self, hooks: Union[Dict, List[Dict]] = None) -> List[Hook]:
        """Build hooks from config.

        Args:
            hooks_cfg (dict): Config dict of hooks.

        Returns:
            list[Hook]: A list of hooks.
        """
        if hooks is not None:
            if isinstance(hooks, dict):
                hooks = [hooks]
            tmp_hooks = []
            for hook in hooks:
                hook = HOOKS.build(hook)
                tmp_hooks.append(hook)
            hooks = tmp_hooks
        return hooks

    @classmethod
    def from_cfg(cls, cfg: ConfigType) -> 'Runner':
        cfg = copy.deepcopy(cfg)
        runner = cls(
            trainer_cfg=cfg.get('trainer_cfg'),
            model_cfg=cfg['model_cfg'],
            datamodule_cfg=cfg.get('datamodule_cfg'),
            cfg=copy.deepcopy(cfg)
        )

        return runner

    def build_logger(self, loggers: Union[Dict, List[Dict]] = None, default_root_dir='logger'):
        if loggers is not None and loggers:
            if isinstance(loggers, Dict):
                loggers = [loggers]
            tmp_loggers = []
            for logger in loggers:
                if logger.get('save_dir', None) is None:
                    logger['save_dir'] = default_root_dir
                mmengine.mkdir_or_exist(logger['save_dir'])
                tmp_loggers.append(LOGGERS.build(logger))
            loggers = tmp_loggers
        return loggers

    def build_strategy(self, strategy='auto'):
        if isinstance(strategy, str):
            return strategy
        elif isinstance(strategy, dict):
            if strategy.get('type', '') == 'FSDPStrategy':
                from torch.distributed.fsdp import CPUOffload
                from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
                import functools
                strategy.update(
                    dict(
                        # cpu_offload=CPUOffload(offload_params=True),
                        auto_wrap_policy=functools.partial(
                            size_based_auto_wrap_policy, min_num_params=int(5e7)
                        )
                    )
                )
            strategy = MODEL_WRAPPERS.build(strategy)
            return strategy
        return strategy

    def build_model(self, model: Union[pl.LightningModule, Dict]) -> pl.LightningModule:
        if isinstance(model, pl.LightningModule):
            return model
        elif isinstance(model, dict):
            model = build_pler(model)
            return model  # type: ignore
        else:
            raise TypeError('model should be a nn.Module object or dict, '
                            f'but got {model}')

    def _init_model_weights(self) -> None:
        """Initialize the model weights if the model has
        :meth:`init_weights`"""
        if hasattr(self.model, 'module'):
            model = self.model.module
        else:
            model = self.model
        if hasattr(model, 'init_weights'):
            model.init_weights()
            # sync params and buffers
            for name, params in model.state_dict().items():
                broadcast(params)

    def get_hooks_info(self) -> str:
        # Get hooks info in each stage
        stage_hook_map: Dict[str, list] = {stage: [] for stage in Hook.stages}
        for hook in self.hooks:
            try:
                priority = Priority(hook.priority).name  # type: ignore
            except ValueError:
                priority = hook.priority  # type: ignore
            classname = hook.__class__.__name__
            hook_info = f'({priority:<12}) {classname:<35}'
            for trigger_stage in hook.get_triggered_stages():
                stage_hook_map[trigger_stage].append(hook_info)

        stage_hook_infos = []
        for stage in Hook.stages:
            hook_infos = stage_hook_map[stage]
            if len(hook_infos) > 0:
                info = f'{stage}:\n'
                info += '\n'.join(hook_infos)
                info += '\n -------------------- '
                stage_hook_infos.append(info)
        return '\n'.join(stage_hook_infos)

    def load_or_resume(self) -> None:
        """load or resume checkpoint."""
        if self._has_loaded:
            return None

        # decide to load from checkpoint or resume from checkpoint
        resume_from = None
        if self._resume and self._load_from is None:
            # auto resume from the latest checkpoint
            resume_from = find_latest_checkpoint(self.work_dir)
            self.logger.info(
                f'Auto resumed from the latest checkpoint {resume_from}.')
        elif self._resume and self._load_from is not None:
            # resume from the specified checkpoint
            resume_from = self._load_from

        if resume_from is not None:
            self.resume(resume_from)
            self._has_loaded = True
        elif self._load_from is not None:
            self.load_checkpoint(self._load_from)
            self._has_loaded = True

    @staticmethod
    def build_datamodule(datamodule_cfg: Union[pl.LightningDataModule, Dict]):
        if isinstance(datamodule_cfg, pl.LightningDataModule):
            return datamodule_cfg
        datamodule_cfg = copy.deepcopy(datamodule_cfg)
        # build datamodule
        datamodule = DATASETS.build(datamodule_cfg)
        return datamodule

    def run(self, status, *args, **kwargs):
        assert status in ['fit', 'test', 'predict', 'validate']
        trainer_func = self.trainer.__getattribute__(status)
        self.datamodule = self.build_datamodule(self.datamodule_cfg)
        return trainer_func(model=self.model, datamodule=self.datamodule, *args, **kwargs)

        #
        # if is_model_wrapper(self.model):
        #     ori_model = self.model.module
        # else:
        #     ori_model = self.model
        # assert hasattr(ori_model, 'train_step'), (
        #     'If you want to train your model, please make sure your model '
        #     'has implemented `train_step`.')
        #
        # if self._val_loop is not None:
        #     assert hasattr(ori_model, 'val_step'), (
        #         'If you want to validate your model, please make sure your '
        #         'model has implemented `val_step`.')
        #
        # if self._train_loop is None:
        #     raise RuntimeError(
        #         '`self._train_loop` should not be None when calling train '
        #         'method. Please provide `train_dataloader`, `train_cfg`, '
        #         '`optimizer` and `param_scheduler` arguments when '
        #         'initializing runner.')
        #
        # self._train_loop = self.build_train_loop(
        #     self._train_loop)  # type: ignore
        #
        # # `build_optimizer` should be called before `build_param_scheduler`
        # #  because the latter depends on the former
        # self.optim_wrapper = self.build_optim_wrapper(self.optim_wrapper)
        # # Automatically scaling lr by linear scaling rule
        # self.scale_lr(self.optim_wrapper, self.auto_scale_lr)
        #
        # if self.param_schedulers is not None:
        #     self.param_schedulers = self.build_param_scheduler(  # type: ignore
        #         self.param_schedulers)  # type: ignore
        #
        # if self._val_loop is not None:
        #     self._val_loop = self.build_val_loop(
        #         self._val_loop)  # type: ignore
        # # TODO: add a contextmanager to avoid calling `before_run` many times
        # self.call_hook('before_run')
        #
        # # initialize the model weights
        # self._init_model_weights()
        # # make sure checkpoint-related hooks are triggered after `before_run`
        # self.load_or_resume()
        #
        # # Initiate inner count of `optim_wrapper`.
        # self.optim_wrapper.initialize_count_status(
        #     self.model,
        #     self._train_loop.iter,  # type: ignore
        #     self._train_loop.max_iters)  # type: ignore
        #
        # # Maybe compile the model according to options in self.cfg.compile
        # # This must be called **AFTER** model has been wrapped.
        # self._maybe_compile('train_step')
        #
        # model = self.train_loop.run()  # type: ignore
        # self.call_hook('after_run')
        # return model



    def register_hook(
            self,
            hook: Union[Hook, Dict],
            priority: Optional[Union[str, int, Priority]] = None) -> None:
        """Register a hook into the hook list.

        The hook will be inserted into a priority queue, with the specified
        priority (See :class:`Priority` for details of priorities).
        For hooks with the same priority, they will be triggered in the same
        order as they are registered.

        Priority of hook will be decided with the following priority:

        - ``priority`` argument. If ``priority`` is given, it will be priority
          of hook.
        - If ``hook`` argument is a dict and ``priority`` in it, the priority
          will be the value of ``hook['priority']``.
        - If ``hook`` argument is a dict but ``priority`` not in it or ``hook``
          is an instance of ``hook``, the priority will be ``hook.priority``.

        Args:
            hook (:obj:`Hook` or dict): The hook to be registered.
            priority (int or str or :obj:`Priority`, optional): Hook priority.
                Lower value means higher priority.
        """
        if not isinstance(hook, (Hook, dict)):
            raise TypeError(
                f'hook should be an instance of Hook or dict, but got {hook}')

        _priority = None
        if isinstance(hook, dict):
            if 'priority' in hook:
                _priority = hook.pop('priority')

            hook_obj = HOOKS.build(hook)
        else:
            hook_obj = hook

        if priority is not None:
            hook_obj.priority = priority
        elif _priority is not None:
            hook_obj.priority = _priority

        inserted = False
        for i in range(len(self._hooks) - 1, -1, -1):
            if get_priority(hook_obj.priority) >= get_priority(
                    self._hooks[i].priority):
                self._hooks.insert(i + 1, hook_obj)
                inserted = True
                break
        if not inserted:
            self._hooks.insert(0, hook_obj)

    def register_default_hooks(
            self,
            hooks: Optional[Dict[str, Union[Hook, Dict]]] = None) -> None:
        """Register default hooks into hook list.

        ``hooks`` will be registered into runner to execute some default
        actions like updating model parameters or saving checkpoints.

        Default hooks and their priorities:

        +----------------------+-------------------------+
        | Hooks                | Priority                |
        +======================+=========================+
        | RuntimeInfoHook      | VERY_HIGH (10)          |
        +----------------------+-------------------------+
        | IterTimerHook        | NORMAL (50)             |
        +----------------------+-------------------------+
        | DistSamplerSeedHook  | NORMAL (50)             |
        +----------------------+-------------------------+
        | LoggerHook           | BELOW_NORMAL (60)       |
        +----------------------+-------------------------+
        | ParamSchedulerHook   | LOW (70)                |
        +----------------------+-------------------------+
        | CheckpointHook       | VERY_LOW (90)           |
        +----------------------+-------------------------+

        If ``hooks`` is None, above hooks will be registered by
        default::

            default_hooks = dict(
                runtime_info=dict(type='RuntimeInfoHook'),
                timer=dict(type='IterTimerHook'),
                sampler_seed=dict(type='DistSamplerSeedHook'),
                logger=dict(type='LoggerHook'),
                param_scheduler=dict(type='ParamSchedulerHook'),
                checkpoint=dict(type='CheckpointHook', interval=1),
            )

        If not None, ``hooks`` will be merged into ``default_hooks``.
        If there are None value in default_hooks, the corresponding item will
        be popped from ``default_hooks``::

            hooks = dict(timer=None)

        The final registered default hooks will be :obj:`RuntimeInfoHook`,
        :obj:`DistSamplerSeedHook`, :obj:`LoggerHook`,
        :obj:`ParamSchedulerHook` and :obj:`CheckpointHook`.

        Args:
            hooks (dict[str, Hook or dict], optional): Default hooks or configs
                to be registered.
        """
        default_hooks: dict = dict(
            runtime_info=dict(type='RuntimeInfoHook'),
            timer=dict(type='IterTimerHook'),
            sampler_seed=dict(type='DistSamplerSeedHook'),
            logger=dict(type='LoggerHook'),
            param_scheduler=dict(type='ParamSchedulerHook'),
            checkpoint=dict(type='CheckpointHook', interval=1),
        )
        if hooks is not None:
            for name, hook in hooks.items():
                if name in default_hooks and hook is None:
                    # remove hook from _default_hooks
                    default_hooks.pop(name)
                else:
                    assert hook is not None
                    default_hooks[name] = hook

        for hook in default_hooks.values():
            self.register_hook(hook)

    def register_custom_hooks(self, hooks: List[Union[Hook, Dict]]) -> None:
        """Register custom hooks into hook list.

        Args:
            hooks (list[Hook | dict]): List of hooks or configs to be
                registered.
        """
        for hook in hooks:
            self.register_hook(hook)

    def register_hooks(
            self,
            default_hooks: Optional[Dict[str, Union[Hook, Dict]]] = None,
            custom_hooks: Optional[List[Union[Hook, Dict]]] = None) -> None:
        """Register default hooks and custom hooks into hook list.

        Args:
            default_hooks (dict[str, dict] or dict[str, Hook], optional): Hooks
                to execute default actions like updating model parameters and
                saving checkpoints.  Defaults to None.
            custom_hooks (list[dict] or list[Hook], optional): Hooks to execute
                custom actions like visualizing images processed by pipeline.
                Defaults to None.
        """
        self.register_default_hooks(default_hooks)

        if custom_hooks is not None:
            self.register_custom_hooks(custom_hooks)

    def resume(self,
               filename: str,
               resume_optimizer: bool = True,
               resume_param_scheduler: bool = True,
               map_location: Union[str, Callable] = 'default') -> None:
        """Resume model from checkpoint.

        Args:
            filename (str): Accept local filepath, URL, ``torchvision://xxx``,
                ``open-mmlab://xxx``.
            resume_optimizer (bool): Whether to resume optimizer state.
                Defaults to True.
            resume_param_scheduler (bool): Whether to resume param scheduler
                state. Defaults to True.
            map_location (str or callable):A string or a callable function to
                specifying how to remap storage locations.
                Defaults to 'default'.
        """
        if map_location == 'default':
            device = get_device()
            checkpoint = self.load_checkpoint(filename, map_location=device)
        else:
            checkpoint = self.load_checkpoint(
                filename, map_location=map_location)

        self.train_loop._epoch = checkpoint['meta']['epoch']
        self.train_loop._iter = checkpoint['meta']['iter']

        # check whether the number of GPU used for current experiment
        # is consistent with resuming from checkpoint
        if 'config' in checkpoint['meta']:
            config = mmengine.Config.fromstring(
                checkpoint['meta']['config'], file_format='.py')
            previous_gpu_ids = config.get('gpu_ids', None)
            if (previous_gpu_ids is not None and len(previous_gpu_ids) > 0
                    and len(previous_gpu_ids) != self._world_size):
                # TODO, should we modify the iteration?
                self.logger.info(
                    'Number of GPU used for current experiment is not '
                    'consistent with resuming from checkpoint')
                if (self.auto_scale_lr is None
                        or not self.auto_scale_lr.get('enable', False)):
                    raise RuntimeError(
                        'Cannot automatically rescale lr in resuming. Please '
                        'make sure the number of GPU is consistent with the '
                        'previous training state resuming from the checkpoint '
                        'or set `enable` in `auto_scale_lr to False.')

        # resume random seed
        resumed_seed = checkpoint['meta'].get('seed', None)
        current_seed = self._randomness_cfg.get('seed')
        if resumed_seed is not None and resumed_seed != current_seed:
            if current_seed is not None:
                print_log(
                    f'The value of random seed in the '
                    f'checkpoint "{resumed_seed}" is '
                    f'different from the value in '
                    f'`randomness` config "{current_seed}"',
                    logger='current',
                    level=logging.WARNING)
            self._randomness_cfg.update(seed=resumed_seed)
            self.set_randomness(**self._randomness_cfg)

        resumed_dataset_meta = checkpoint['meta'].get('dataset_meta', None)
        dataset_meta = getattr(self.train_dataloader.dataset, 'metainfo', None)

        # `resumed_dataset_meta` and `dataset_meta` could be object like
        # np.ndarray, which cannot be directly judged as equal or not,
        # therefore we just compared their dumped results.
        if pickle.dumps(resumed_dataset_meta) != pickle.dumps(dataset_meta):
            print_log(
                'The dataset metainfo from the resumed checkpoint is '
                'different from the current training dataset, please '
                'check the correctness of the checkpoint or the training '
                'dataset.',
                logger='current',
                level=logging.WARNING)

        self.message_hub.load_state_dict(checkpoint['message_hub'])

        # resume optimizer
        if 'optimizer' in checkpoint and resume_optimizer:
            self.optim_wrapper = self.build_optim_wrapper(self.optim_wrapper)
            self.optim_wrapper.load_state_dict(  # type: ignore
                checkpoint['optimizer'])

        # resume param scheduler
        if resume_param_scheduler and self.param_schedulers is None:
            print_log(
                '`resume_param_scheduler` is True but `self.param_schedulers` '
                'is None, so skip resuming parameter schedulers',
                logger='current',
                level=logging.WARNING)
            resume_param_scheduler = False
        if 'param_schedulers' in checkpoint and resume_param_scheduler:
            self.param_schedulers = self.build_param_scheduler(  # type: ignore
                self.param_schedulers)  # type: ignore
            if isinstance(self.param_schedulers, dict):
                for name, schedulers in self.param_schedulers.items():
                    for scheduler, ckpt_scheduler in zip(
                            schedulers, checkpoint['param_schedulers'][name]):
                        scheduler.load_state_dict(ckpt_scheduler)
            else:
                for scheduler, ckpt_scheduler in zip(
                        self.param_schedulers,  # type: ignore
                        checkpoint['param_schedulers']):
                    scheduler.load_state_dict(ckpt_scheduler)

        self._has_loaded = True

        self.logger.info(f'resumed epoch: {self.epoch}, iter: {self.iter}')

    # def load_checkpoint(self,
    #                     filename: str,
    #                     model,
    #                     map_location: Union[str, Callable] = 'cpu',
    #                     strict: bool = False,
    #                     revise_keys: list = [(r'^module.', '')]):
    #     """Load checkpoint from given ``filename``.
    #
    #     Args:
    #         filename (str): Accept local filepath, URL, ``torchvision://xxx``,
    #             ``open-mmlab://xxx``.
    #         map_location (str or callable): A string or a callable function to
    #             specifying how to remap storage locations.
    #             Defaults to 'cpu'.
    #         strict (bool): strict (bool): Whether to allow different params for
    #             the model and checkpoint.
    #         revise_keys (list): A list of customized keywords to modify the
    #             state_dict in checkpoint. Each item is a (pattern, replacement)
    #             pair of the regular expression operations. Defaults to strip
    #             the prefix 'module.' by [(r'^module\\.', '')].
    #     """
    #     checkpoint = _load_checkpoint(filename, map_location=map_location)
    #
    #     if is_model_wrapper(model):
    #         model = model.module
    #     else:
    #         model = model
    #
    #     checkpoint = _load_checkpoint_to_model(
    #         model, checkpoint, strict, revise_keys=revise_keys)
    #
    #     print(f'Load checkpoint from {filename}')
    #
    #     return checkpoint
    def load_checkpoint(self, model, file):

        if isinstance(file, str):
            file_path = file
            state_dict = torch.load(file_path, map_location='cpu')['state_dict']
        elif isinstance(file, dict):
            file_path = file['file_path']
            state_dict = torch.load(file_path, map_location='cpu')['state_dict']
            for delete_key in file['delete_keys']:
                del state_dict[delete_key]
        else:
            raise TypeError('file must be str or dict')
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print('load from:', file_path)
        print('load model missing_keys:', missing_keys)
        print('load model unexpected_keys:', unexpected_keys)

    @master_only
    def save_checkpoint(
        self,
        out_dir: str,
        filename: str,
        file_client_args: Optional[dict] = None,
        save_optimizer: bool = True,
        save_param_scheduler: bool = True,
        meta: dict = None,
        by_epoch: bool = True,
        backend_args: Optional[dict] = None,
    ):
        """Save checkpoints.

        ``CheckpointHook`` invokes this method to save checkpoints
        periodically.

        Args:
            out_dir (str): The directory that checkpoints are saved.
            filename (str): The checkpoint filename.
            file_client_args (dict, optional): Arguments to instantiate a
                FileClient. See :class:`mmengine.fileio.FileClient` for
                details. Defaults to None. It will be deprecated in future.
                Please use `backend_args` instead.
            save_optimizer (bool): Whether to save the optimizer to
                the checkpoint. Defaults to True.
            save_param_scheduler (bool): Whether to save the param_scheduler
                to the checkpoint. Defaults to True.
            meta (dict, optional): The meta information to be saved in the
                checkpoint. Defaults to None.
            by_epoch (bool): Whether the scheduled momentum is updated by
                epochs. Defaults to True.
            backend_args (dict, optional): Arguments to instantiate the
                prefix of uri corresponding backend. Defaults to None.
                New in v0.2.0.
        """
        if meta is None:
            meta = {}
        elif not isinstance(meta, dict):
            raise TypeError(
                f'meta should be a dict or None, but got {type(meta)}')

        if by_epoch:
            # self.epoch increments 1 after
            # `self.call_hook('after_train_epoch)` but `save_checkpoint` is
            # called by `after_train_epoch`` method of `CheckpointHook` so
            # `epoch` should be `self.epoch + 1`
            meta.update(epoch=self.epoch + 1, iter=self.iter)
        else:
            meta.update(epoch=self.epoch, iter=self.iter + 1)

        if file_client_args is not None:
            warnings.warn(
                '"file_client_args" will be deprecated in future. '
                'Please use "backend_args" instead', DeprecationWarning)
            if backend_args is not None:
                raise ValueError(
                    '"file_client_args" and "backend_args" cannot be set at '
                    'the same time.')

            file_client = FileClient.infer_client(file_client_args, out_dir)
            filepath = file_client.join_path(out_dir, filename)
        else:
            filepath = join_path(  # type: ignore
                out_dir, filename, backend_args=backend_args)

        meta.update(
            cfg=self.cfg.pretty_text,
            seed=self.seed,
            experiment_name=self.experiment_name,
            time=time.strftime('%Y%m%d_%H%M%S', time.localtime()),
            mmengine_version=mmengine.__version__ + get_git_hash())

        if hasattr(self.train_dataloader.dataset, 'metainfo'):
            meta.update(dataset_meta=self.train_dataloader.dataset.metainfo)

        if is_model_wrapper(self.model):
            model = self.model.module
        else:
            model = self.model

        checkpoint = {
            'meta': meta,
            'state_dict': weights_to_cpu(get_state_dict(model)),
            'message_hub': self.message_hub.state_dict()
        }
        # save optimizer state dict to checkpoint
        if save_optimizer:
            if isinstance(self.optim_wrapper, OptimWrapper):
                checkpoint['optimizer'] = self.optim_wrapper.state_dict()
            else:
                raise TypeError(
                    'self.optim_wrapper should be an `OptimWrapper` '
                    'or `OptimWrapperDict` instance, but got '
                    f'{self.optim_wrapper}')

        # save param scheduler state dict
        if save_param_scheduler and self.param_schedulers is None:
            print_log(
                '`save_param_scheduler` is True but `self.param_schedulers` '
                'is None, so skip saving parameter schedulers',
                logger='current',
                level=logging.WARNING)
            save_param_scheduler = False
        if save_param_scheduler:
            if isinstance(self.param_schedulers, dict):
                checkpoint['param_schedulers'] = dict()
                for name, schedulers in self.param_schedulers.items():
                    checkpoint['param_schedulers'][name] = []
                    for scheduler in schedulers:
                        state_dict = scheduler.state_dict()
                        checkpoint['param_schedulers'][name].append(state_dict)
            else:
                checkpoint['param_schedulers'] = []
                for scheduler in self.param_schedulers:  # type: ignore
                    state_dict = scheduler.state_dict()  # type: ignore
                    checkpoint['param_schedulers'].append(state_dict)

        self.call_hook('before_save_checkpoint', checkpoint=checkpoint)
        save_checkpoint(checkpoint, filepath)

    @master_only
    def dump_config(self) -> None:
        version = ''
        if len(self.trainer.loggers) > 0:
            version = self.trainer.loggers[0].version
            version = version if isinstance(version, str) else f"version_{version}"
        if version == '':
            # if no loggers, use default_root_dir
            version = 'version'

        """Dump config to `work_dir`."""
        if self.cfg.filename is not None:
            filename = osp.basename(self.cfg.filename)
        else:
            filename = f'{self.timestamp}.py'
        path = f'{self.trainer.default_root_dir}/{version}_{filename}'

        self.cfg.dump(path)

    def _check_scheduler_cfg(
            self, param_scheduler: Optional[Union[dict, list,
                                                  _ParamScheduler]]) -> None:
        """Parse `param_scheduler` to a list of parameter schedulers, or a
        `dict` of which each value is a list of parameter schedulers.

        If only one optimizer is used, the parsed config should be a
        list of parameter scheduler configs or instances. If multiple
        optimizers are used, the parsed config should be `dict`.
        Its key should be consistent with the optimizer `dict` and its value
        should be a list of parameter scheduler configs or instances. See
        :meth:`build_param_scheduler` for more details.

        Examples:
            >>> # valid scheduler:
            >>> # empty scheduler
            >>> scheduler = None
            >>> # Single scheduler
            >>> scheduler = dict(type='MultiStepLR', milestones=[1, 2])
            >>> # Single list schedulers
            >>> scheduler = [dict(type='MultiStepLR', milestones=[1, 2]),
            >>>              dict(type='MultiStepLR', milestones=[2, 3])]
            >>> # `dict` of schedulers
            >>> scheduler = dict(linear1=dict(type='MultiStepLR', milestones=[1, 2]),
            >>>                  linear2=dict(type='MultiStepLR', milestones=[1, 2]))
            >>> # `dict` of `list` of schedulers
            >>> scheduler = dict(linear1=[dict(type='MultiStepLR', milestones=[1, 2])],
            >>>                  linear2=[dict(type='MultiStepLR', milestones=[1, 2])])
            >>> # Single built scheduler
            >>> from mmengine.optim import MultiStepLR
            >>> scheduler = MultiStepLR(milestones=[1, 2], optimizer=optimizer)
            >>> # Single built list schedulers
            >>> scheduler = [MultiStepLR(milestones=[1, 2], optimizer=optimizer)]
            >>> # dict of built scheduler
            >>> scheduler = dict(linear1=MultiStepLR(milestones=[1, 2], optimizer=optimizer),
            >>>                  linear2=MultiStepLR(milestones=[1, 2], optimizer=optimizer))
            >>> # dict of built list schedulers
            >>> scheduler = dict(linear1=[MultiStepLR(milestones=[1, 2], optimizer=optimizer)],
            >>>                  linear2=[MultiStepLR(milestones=[1, 2], optimizer=optimizer)])

        Args:
            param_scheduler (dict or list): The original parameter scheduler.
        """  # noqa: E501
        param_schedulers: Union[dict, list, _ParamScheduler]
        if param_scheduler is None:
            return
        if isinstance(param_scheduler, _ParamScheduler):
            return
        if is_seq_of(param_scheduler, _ParamScheduler):
            return

        if is_seq_of(param_scheduler, dict):
            for _param_scheduler in param_scheduler:
                assert 'type' in _param_scheduler, (
                    'Each parameter scheduler should contain the key type, '
                    f'but got {_param_scheduler}')
        elif isinstance(param_scheduler, dict):
            if 'type' not in param_scheduler:
                for key, _param_scheduler in param_scheduler.items():
                    assert isinstance(
                        _param_scheduler,
                        (dict, tuple, list, _ParamScheduler)), (
                            'Each value of `param_scheduler` should be a '
                            f'dict or a list, but got {_param_scheduler} with '
                            f'type {type(_ParamScheduler)}')

        else:
            raise TypeError(
                '`param_scheduler` should be a `_ParamScheduler`, `dict`, '
                f'list or a tuple, but got {type(param_scheduler)}. If '
                '`param_scheduler` is a list of dict, it means a list of '
                'scheduler configs for single optimizer. If it is a dict and '
                'contains key `type`, it means a scheduler config for a '
                'single optimizer. If it does not contain key `type`, it '
                'means multiple lists of schedulers for multiple optimizers.')

    def _log_env(self, env_cfg: dict) -> None:
        """Logging environment information of the current task.

        Args:
            env_cfg (dict): The environment config of the runner.
        """
        # Collect and log environment information.
        env = collect_env()
        runtime_env = OrderedDict()
        runtime_env.update(env_cfg)
        runtime_env.update(self._randomness_cfg)
        runtime_env['Distributed launcher'] = self._launcher
        runtime_env['Distributed training'] = self._distributed
        runtime_env['GPU number'] = self._world_size

        env_info = '\n    ' + '\n    '.join(f'{k}: {v}'
                                            for k, v in env.items())
        runtime_env_info = '\n    ' + '\n    '.join(
            f'{k}: {v}' for k, v in runtime_env.items())
        dash_line = '-' * 60
        self.logger.info('\n' + dash_line + '\nSystem environment:' +
                         env_info + '\n'
                         '\nRuntime environment:' + runtime_env_info + '\n' +
                         dash_line + '\n')
        self.logger.info(f'Config:\n{self.cfg.pretty_text}')

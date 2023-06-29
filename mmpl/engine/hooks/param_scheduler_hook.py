from typing import Dict, Optional, Union, Any

from lightning.pytorch.utilities.types import STEP_OUTPUT
from mmengine.optim import _ParamScheduler
from mmpl.registry import HOOKS
from mmengine.utils import is_list_of
from lightning import Callback

DATA_BATCH = Optional[Union[dict, tuple, list]]


@HOOKS.register_module()
class ParamSchedulerHook(Callback):
    """A hook to update some hyper-parameters in optimizer, e.g., learning rate
    and momentum."""

    priority = 'LOW'

    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        """Call step function for each scheduler after each training iteration.

        Args:
            runner (Runner): The runner of the training process.
            batch_idx (int): The index of the current batch in the train loop.
            data_batch (dict or tuple or list, optional): Data from dataloader.
                In order to keep this interface consistent with other hooks,
                we keep ``data_batch`` here.
            outputs (dict, optional): Outputs from model.
                In order to keep this interface consistent with other hooks, we
                keep ``data_batch`` here.
        """
        param_schedulers = pl_module.lr_schedulers()
        if param_schedulers is None:
            return

        def step(param_schedulers):
            assert isinstance(param_schedulers, list)
            for scheduler in param_schedulers:
                if not scheduler.by_epoch:
                    scheduler.step()
        if isinstance(param_schedulers, _ParamScheduler):
            param_schedulers = [param_schedulers]
        if isinstance(param_schedulers, list):
            step(param_schedulers)
        elif isinstance(param_schedulers, dict):
            for param_schedulers in param_schedulers.values():
                step(param_schedulers)
        else:
            raise TypeError(
                'runner.param_schedulers should be list of ParamScheduler or '
                'a dict containing list of ParamScheduler, '
                f'but got {param_schedulers}')

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Call step function for each scheduler after each training epoch.

        Args:
            runner (Runner): The runner of the training process.
        """
        param_schedulers = pl_module.lr_schedulers()
        if param_schedulers is None:
            return

        def step(param_schedulers):
            assert isinstance(param_schedulers, list)
            for scheduler in param_schedulers:
                if scheduler.by_epoch:
                    scheduler.step()
        if isinstance(param_schedulers, _ParamScheduler):
            param_schedulers = [param_schedulers]
        if isinstance(param_schedulers, list):
            step(param_schedulers)
        elif isinstance(param_schedulers, dict):
            for param_schedulers in param_schedulers.values():
                step(param_schedulers)
        else:
            raise TypeError(
                'runner.param_schedulers should be list of ParamScheduler or '
                'a dict containing list of ParamScheduler, '
                f'but got {param_schedulers}')

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Call step function for each scheduler which has attribute
        ``need_val_args`` after each validation epoch.

        Args:
            runner (Runner): The runner of the validation process.
            metrics (Dict[str, float], optional): Evaluation results of all
                metrics on validation dataset. The keys are the names of the
                metrics, and the values are corresponding results.

        Note:
            if ``runner.param_schedulers`` is not built before,
            the hook ``after_val_epoch`` will be skipped.
        """
        param_schedulers = pl_module.lr_schedulers()
        if param_schedulers is None:
            return

        # avoid counting scheduler._global_step
        # it has counted in after_train_* hook
        metrics = trainer.callback_metrics
        if metrics is None:
            return

        def step(param_schedulers):
            # check param_schedulers is list and built
            if not is_list_of(param_schedulers, _ParamScheduler):
                return

            for scheduler in param_schedulers:
                if (scheduler.by_epoch
                        and getattr(scheduler, 'need_val_args', False)):
                    scheduler.step(metrics)
        if isinstance(param_schedulers, _ParamScheduler):
            param_schedulers = [param_schedulers]
        if isinstance(param_schedulers, list):
            step(param_schedulers)
        elif isinstance(param_schedulers, dict):
            for param_schedulers in param_schedulers.values():
                step(param_schedulers)
        else:
            raise TypeError(
                'runner.param_schedulers should be list of ParamScheduler or '
                'a dict containing list of ParamScheduler, '
                f'but got {param_schedulers}')

from mmcv.transforms import Compose
from mmpl.registry import HOOKS
from lightning.pytorch.callbacks import Callback


@HOOKS.register_module()
class PipelineSwitchHook(Callback):
    """Switch data pipeline at switch_epoch.

    Args:
        switch_epoch (int): switch pipeline at this epoch.
        switch_pipeline (list[dict]): the pipeline to switch to.
    """

    def __init__(self, switch_epoch, switch_pipeline):
        self.switch_epoch = switch_epoch
        self.switch_pipeline = switch_pipeline
        self._restart_dataloader = False

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """switch pipeline."""
        epoch = trainer.current_epoch
        train_loader = trainer.train_dataloader
        if epoch == self.switch_epoch:
            if trainer.local_rank == 0:
                print('Switch pipeline now!')
            # The dataset pipeline cannot be updated when persistent_workers
            # is True, so we need to force the dataloader's multi-process
            # restart. This is a very hacky approach.
            train_loader.dataset.pipeline = Compose(self.switch_pipeline)
            if hasattr(train_loader, 'persistent_workers'
                       ) and train_loader.persistent_workers is True:
                train_loader._DataLoader__initialized = False
                train_loader._iterator = None
                self._restart_dataloader = True

        else:
            # Once the restart is complete, we need to restore
            # the initialization flag.
            if self._restart_dataloader:
                train_loader._DataLoader__initialized = True

import copy
import inspect
from typing import List, Union

import torch
import torch.nn as nn
import lightning

from mmengine.config import Config, ConfigDict
from mmengine.device import is_npu_available
from mmpl.registry import HOOKS


def register_pl_hooks() -> List[str]:
    """Register callbacks in ``lightning.pytorch.callbacks`` to the ``HOOKS`` registry.

    Returns:
        List[str]: A list of registered callbacks' name.
    """
    pl_hooks = []
    for module_name in dir(lightning.pytorch.callbacks):
        if module_name.startswith('__'):
            continue
        _hook = getattr(lightning.pytorch.callbacks, module_name)
        if inspect.isclass(_hook) and issubclass(_hook, lightning.pytorch.callbacks.Callback):
            HOOKS.register_module(module=_hook)
            pl_hooks.append(module_name)
    return pl_hooks


PL_HOOKS = register_pl_hooks()

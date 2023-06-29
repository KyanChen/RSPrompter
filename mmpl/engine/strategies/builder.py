import inspect
from typing import List, Union

import torch
import lightning
from mmpl.registry import MODEL_WRAPPERS


def register_pl_strategies() -> List[str]:
    """Register callbacks in ``lightning.pytorch.callbacks`` to the ``HOOKS`` registry.

    Returns:
        List[str]: A list of registered callbacks' name.
    """
    pl_strategies = []
    for module_name in dir(lightning.pytorch.strategies):
        if module_name.startswith('__'):
            continue
        _strategy = getattr(lightning.pytorch.strategies, module_name)
        if inspect.isclass(_strategy) and issubclass(_strategy, lightning.pytorch.strategies.Strategy):
            MODEL_WRAPPERS.register_module(module=_strategy)
            pl_strategies.append(module_name)
    return pl_strategies


PL_MODEL_WRAPPERS = register_pl_strategies()

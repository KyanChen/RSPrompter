# Copyright (c) OpenMMLab. All rights reserved.
from .collect_env import collect_env
from .misc import is_metainfo_lower, switch_to_deploy
from .setup_env import register_all_modules
from .typing_utils import *

__all__ = [
    'register_all_modules', 'collect_env', 'switch_to_deploy',
    'is_metainfo_lower', 'ConfigType', 'OptMultiConfig', 'MultiConfig',
]

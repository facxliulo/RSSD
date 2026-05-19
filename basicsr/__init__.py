# basicsr/__init__.py
# 避免重复导入导致的注册冲突

# 版本信息
__version__ = '1.0.0'

# 只导入必要的模块，避免循环导入
from .utils import *  # 先导入工具函数

# 延迟导入其他模块，避免重复注册
def _lazy_import():
    """延迟导入，避免启动时的注册冲突"""
    from . import archs
    from . import data
    from . import metrics
    from . import models
    from . import options

# 暴露构建函数
from .data import build_dataset, build_dataloader
from .models import build_model
from .archs import build_network

__all__ = [
    '__version__',
    'build_dataset',
    'build_dataloader', 
    'build_model',
    'build_network',
]

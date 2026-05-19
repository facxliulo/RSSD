# basicsr/data/__init__.py 详细注释

# 导入必要的库和模块
import importlib  # 用于动态导入模块
import numpy as np
import random
import torch
import torch.utils.data  # PyTorch数据加载相关工具
from copy import deepcopy  # 深度拷贝，避免修改原始配置
from functools import partial  # 用于创建偏函数
from os import path as osp  # 操作系统路径处理

# 导入自定义模块
from new_basicsr.data.prefetch_dataloader import PrefetchDataLoader  # 预取数据加载器
from new_basicsr.utils import get_root_logger, scandir  # 获取根日志记录器和文件扫描工具
from new_basicsr.utils.dist_util import get_dist_info  # 分布式训练信息获取
from new_basicsr.utils.registry import DATASET_REGISTRY  # 数据集注册表
from new_basicsr.data.semantic_mask_dataset_flexible import FlexibleSemanticMaskDataset

# 定义模块的公共接口
__all__ = ['build_dataset', 'build_dataloader']

# 自动扫描并导入数据集模块到注册表中
# 扫描data文件夹下所有以'_dataset.py'结尾的文件
data_folder = osp.dirname(osp.abspath(__file__))  # 获取当前文件所在目录
dataset_filenames = [
    osp.splitext(osp.basename(v))[0]  # 获取不带扩展名的文件名
    for v in scandir(data_folder)  # 扫描目录中的所有文件
    if v.endswith('_dataset.py')  # 只处理以'_dataset.py'结尾的文件
]
# 导入所有数据集模块
_dataset_modules = [
    importlib.import_module(f'new_basicsr.data.{file_name}')  # 动态导入模块
    for file_name in dataset_filenames
]


def build_dataset(dataset_opt):
    """根据配置选项构建数据集。

    参数:
        dataset_opt (dict): 数据集的配置选项。必须包含:
            name (str): 数据集名称。
            type (str): 数据集类型。

    返回:
        Dataset: 构建的数据集实例
    """
    # 深度拷贝配置选项，避免修改原始配置
    dataset_opt = deepcopy(dataset_opt)
    # 从注册表中获取对应类型的数据集类并实例化
    dataset = DATASET_REGISTRY.get(dataset_opt['type'])(dataset_opt)
    # 获取根日志记录器
    logger = get_root_logger()
    # 记录数据集构建信息
    logger.info(f'Dataset [{dataset.__class__.__name__}] - {dataset_opt["name"]} is built.')
    return dataset


def build_dataloader(dataset, dataset_opt, num_gpu=1, dist=False, sampler=None, seed=None):
    """构建数据加载器。

    参数:
        dataset (torch.utils.data.Dataset): 数据集实例。
        dataset_opt (dict): 数据集选项。包含以下键:
            phase (str): 'train' 或 'val'，表示训练或验证阶段。
            num_worker_per_gpu (int): 每个GPU的工作进程数。
            batch_size_per_gpu (int): 每个GPU的训练批次大小。
        num_gpu (int): GPU数量。仅在训练阶段使用。默认: 1。
        dist (bool): 是否在分布式训练中。仅在训练阶段使用。默认: False。
        sampler (torch.utils.data.sampler): 数据采样器。默认: None。
        seed (int | None): 随机种子。默认: None。

    返回:
        DataLoader: 构建的数据加载器
    """
    # 获取当前阶段（训练或验证/测试）
    phase = dataset_opt['phase']
    # 获取分布式训练中的进程排名
    rank, _ = get_dist_info()

    # 训练阶段的数据加载器配置
    if phase == 'train':
        if dist:  # 分布式训练
            # 每个GPU的批次大小
            batch_size = dataset_opt['batch_size_per_gpu']
            # 每个GPU的工作进程数
            num_workers = dataset_opt['num_worker_per_gpu']
        else:  # 非分布式训练
            # 计算倍增因子（GPU数量）
            multiplier = 1 if num_gpu == 0 else num_gpu
            # 总批次大小 = 每个GPU的批次大小 × GPU数量
            batch_size = dataset_opt['batch_size_per_gpu'] * multiplier
            # 总工作进程数 = 每个GPU的工作进程数 × GPU数量
            num_workers = dataset_opt['num_worker_per_gpu'] * multiplier

        # 数据加载器参数配置
        dataloader_args = dict(
            dataset=dataset,  # 数据集
            batch_size=batch_size,  # 批次大小
            shuffle=False,  # 默认不洗牌（如果使用采样器）
            num_workers=num_workers,  # 工作进程数
            sampler=sampler,  # 采样器
            drop_last=True)  # 丢弃最后不完整的批次

        # 如果没有提供采样器，则启用洗牌
        if sampler is None:
            dataloader_args['shuffle'] = True

        # 设置工作进程初始化函数（用于设置随机种子）
        dataloader_args['worker_init_fn'] = partial(
            worker_init_fn, num_workers=num_workers, rank=rank, seed=seed) if seed is not None else None

    # 验证或测试阶段的数据加载器配置
    elif phase in ['val', 'test']:
        dataloader_args = dict(
            dataset=dataset,
            batch_size=1,  # 验证/测试通常使用批次大小为1
            shuffle=False,  # 验证/测试不需要洗牌
            num_workers=0)  # 验证/测试通常不需要多进程加载

    else:
        raise ValueError(f"Wrong dataset phase: {phase}. Supported ones are 'train', 'val' and 'test'.")

    # 设置额外的数据加载器参数
    dataloader_args['pin_memory'] = dataset_opt.get('pin_memory', False)  # 是否固定内存（加速GPU传输）
    dataloader_args['persistent_workers'] = dataset_opt.get('persistent_workers', False)  # 是否保持工作进程

    # 根据预取模式选择不同的数据加载器
    prefetch_mode = dataset_opt.get('prefetch_mode')
    if prefetch_mode == 'cpu':  # 使用CPU预取器
        num_prefetch_queue = dataset_opt.get('num_prefetch_queue', 1)  # 预取队列大小
        logger = get_root_logger()
        logger.info(f'Use {prefetch_mode} prefetch dataloader: num_prefetch_queue = {num_prefetch_queue}')
        # 返回预取数据加载器
        return PrefetchDataLoader(num_prefetch_queue=num_prefetch_queue, **dataloader_args)
    else:
        # prefetch_mode=None: 普通数据加载器
        # prefetch_mode='cuda': 用于CUDAPrefetcher的数据加载器
        return torch.utils.data.DataLoader(**dataloader_args)


def worker_init_fn(worker_id, num_workers, rank, seed):
    """工作进程初始化函数，用于设置每个工作进程的随机种子。

    参数:
        worker_id (int): 工作进程ID。
        num_workers (int): 工作进程总数。
        rank (int): 当前进程在分布式训练中的排名。
        seed (int): 基础随机种子。
    """
    # 计算工作进程特定的种子：num_workers * rank + worker_id + seed
    worker_seed = num_workers * rank + worker_id + seed
    # 设置NumPy和Python随机种子，确保每个进程的数据加载具有可重复性
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def build_combined_dataset(inpaint_opt, sr_opt):
    """构建联合数据集"""
    from .combined_dataset import CombinedDataset
    return CombinedDataset({
        'inpaint': inpaint_opt,
        'sr': sr_opt
    })
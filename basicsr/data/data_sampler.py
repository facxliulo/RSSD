# basicsr/data/data_sampler.py 详细注释

import math
import torch
from torch.utils.data.sampler import Sampler  # 导入PyTorch采样器基类


class EnlargedSampler(Sampler):
    """扩展采样器，限制数据加载到数据集的子集。

    基于torch.utils.data.distributed.DistributedSampler修改
    支持扩展数据集以进行基于迭代的训练，以便在每个epoch后重新启动数据加载器时节省时间

    参数:
        dataset (torch.utils.data.Dataset): 用于采样的数据集。
        num_replicas (int | None): 参与训练的进程数。通常是world_size（分布式训练中的总进程数）。
        rank (int | None): 当前进程在num_replicas中的排名。
        ratio (int): 扩展比率。默认: 1。
    """

    def __init__(self, dataset, num_replicas, rank, ratio=1):
        """
        初始化扩展采样器。

        参数:
            dataset: 要采样的数据集
            num_replicas: 分布式训练中的进程总数
            rank: 当前进程的排名
            ratio: 数据集扩展比率
        """
        # 调用父类构造函数
        super().__init__(dataset)

        # 存储传入的参数
        self.dataset = dataset  # 数据集对象
        self.num_replicas = num_replicas  # 分布式训练中的进程总数
        self.rank = rank  # 当前进程的排名
        self.epoch = 0  # 当前epoch数，用于生成确定性随机排列
        # 计算每个进程应该采样的样本数
        self.num_samples = math.ceil(len(self.dataset) * ratio / self.num_replicas)
        # 计算扩展后的总样本数（所有进程的总和）
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        """创建采样器的迭代器。

        返回:
            一个迭代器，产生当前进程应该处理的样本索引
        """
        # 基于当前epoch生成确定性随机种子，确保每个epoch的采样顺序不同
        # 但在相同epoch和相同随机种子的情况下，采样顺序可重现
        g = torch.Generator()
        g.manual_seed(self.epoch)

        # 生成扩展后总样本数的随机排列
        indices = torch.randperm(self.total_size, generator=g).tolist()

        # 将扩展的索引映射回原始数据集大小
        dataset_size = len(self.dataset)
        indices = [v % dataset_size for v in indices]

        # 子采样：为当前进程选择对应的索引
        # 从rank位置开始，每隔num_replicas取一个索引
        indices = indices[self.rank:self.total_size:self.num_replicas]

        # 确保当前进程采样的索引数量等于预期的num_samples
        assert len(indices) == self.num_samples

        # 返回索引的迭代器
        return iter(indices)

    def __len__(self):
        """返回当前进程应该采样的样本数量。

        返回:
            int: 样本数量
        """
        return self.num_samples

    def set_epoch(self, epoch):
        """设置当前epoch数。

        参数:
            epoch (int): 当前的epoch数
        """
        self.epoch = epoch
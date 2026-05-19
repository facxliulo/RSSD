import torch.utils.data as data
from basicsr.utils.registry import DATASET_REGISTRY
from .inpainting_dataset import InpaintingDataset
from .paired_image_dataset import PairedImageDataset


@DATASET_REGISTRY.register()
class CombinedDataset(data.Dataset):
    """联合数据集：同时包含图像修复和超分数据"""

    def __init__(self, opt):
        super(CombinedDataset, self).__init__()

        # 解析子数据集配置
        inpaint_opt = opt.get('inpaint', {})
        sr_opt = opt.get('sr', {})

        # 初始化子数据集
        self.inpaint_dataset = InpaintingDataset(inpaint_opt)
        self.sr_dataset = PairedImageDataset(sr_opt)

        # 使用较大数据集的长度
        self.length = max(len(self.inpaint_dataset), len(self.sr_dataset))

        print(
            f"联合数据集创建完成 - 修复数据: {len(self.inpaint_dataset)}, 超分数据: {len(self.sr_dataset)}, 总长度: {self.length}")

    def __getitem__(self, index):
        # 获取修复数据
        inpaint_data = self.inpaint_dataset[index % len(self.inpaint_dataset)]

        # 获取超分数据
        sr_data = self.sr_dataset[index % len(self.sr_dataset)]

        # 合并数据
        return {
            'img': inpaint_data['img'],  # 修复用原图
            'mask': inpaint_data['mask'],  # 修复用掩码
            'lq': sr_data['lq'],  # 超分用低清图
            'gt': sr_data['gt'],  # 超分用高清图
            'img_path': inpaint_data['img_path'],
            'gt_path': sr_data['gt_path'],
        }

    def __len__(self):
        return self.length
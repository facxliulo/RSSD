import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class InpaintingDataset(Dataset):
    """图像修复数据集 - 支持多掩码叠加和半透明"""

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        self.image_paths = self._load_flist(opt.get('image_flist', ''))
        self.mask_paths = self._load_flist(opt.get('mask_flist', ''))

        self.img_size = opt.get('img_size', 256)
        
        # 新增配置项
        self.min_masks = opt.get('min_masks', 1)  # 最少掩码数量
        self.max_masks = opt.get('max_masks', 4)  # 最多掩码数量
        self.use_semi_transparent = opt.get('use_semi_transparent', True)  # 是否使用半透明
        self.semi_transparent_prob = opt.get('semi_transparent_prob', 0.3)  # 半透明概率
        self.transparency_range = opt.get('transparency_range', [0.3, 0.7])  # 半透明范围
        
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
        ])

        print(f"图像修复数据集加载完成: {len(self.image_paths)} 张图片, {len(self.mask_paths)} 个掩码")
        print(f"  掩码叠加: {self.min_masks}-{self.max_masks} 张")
        print(f"  半透明: {'启用' if self.use_semi_transparent else '禁用'} (概率: {self.semi_transparent_prob})")

    def _load_flist(self, flist_path):
        """加载文件列表"""
        if not os.path.exists(flist_path):
            print(f"警告: 文件列表不存在: {flist_path}")
            return []

        paths = []
        with open(flist_path, 'r') as f:
            for line in f:
                path = line.strip()
                if path and os.path.exists(path):
                    paths.append(path)
        return paths

    def __len__(self):
        return max(len(self.image_paths), 1)

    def __getitem__(self, index):
        try:
            # 加载图像
            if len(self.image_paths) > 0:
                img_path = self.image_paths[index % len(self.image_paths)]
                image = Image.open(img_path).convert('RGB')
                image = self.transform(image)
            else:
                image = torch.rand(3, self.img_size, self.img_size)

            # 生成复合掩码（1-4张随机叠加）
            if len(self.mask_paths) > 0:
                mask = self._generate_composite_mask()
            else:
                mask = self._generate_random_mask()

            return {
                'img': image,
                'mask': mask,
                'img_path': img_path if len(self.image_paths) > 0 else 'generated',
            }

        except Exception as e:
            print(f"数据加载错误 (index: {index}): {e}")
            return {
                'img': torch.rand(3, self.img_size, self.img_size),
                'mask': self._generate_random_mask(),
                'img_path': 'error_fallback',
            }

    def _generate_composite_mask(self):
        """
        生成复合掩码：随机叠加1-4张掩码
        支持半透明效果（白色及半透明白色表示破损区域）
        """
        # 随机决定叠加几张掩码
        num_masks = random.randint(self.min_masks, self.max_masks)
        
        # 初始化复合掩码（全黑）
        composite_mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        
        for i in range(num_masks):
            # 随机选择一个掩码
            mask_path = self.mask_paths[random.randint(0, len(self.mask_paths) - 1)]
            
            try:
                # 加载掩码
                mask = Image.open(mask_path).convert('L')
                mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
                mask_array = np.array(mask, dtype=np.float32) / 255.0
                
                # 二值化（>0.5为破损区域）
                mask_binary = (mask_array > 0.5).astype(np.float32)
                
                # 是否使用半透明
                if self.use_semi_transparent and random.random() < self.semi_transparent_prob:
                    # 随机透明度（0.3-0.7表示半透明破损）
                    transparency = random.uniform(self.transparency_range[0], self.transparency_range[1])
                    mask_value = transparency
                else:
                    # 完全不透明（1.0表示完全破损）
                    mask_value = 1.0
                
                # 叠加到复合掩码（取最大值，保留最严重的破损）
                composite_mask = np.maximum(composite_mask, mask_binary * mask_value)
                
            except Exception as e:
                print(f"警告: 加载掩码失败 ({mask_path}): {e}")
                continue
        
        # 转换为tensor
        mask_tensor = torch.from_numpy(composite_mask).unsqueeze(0)  # [1, H, W]
        
        return mask_tensor

    def _generate_random_mask(self):
        """生成随机掩码（fallback）"""
        mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        
        # 随机生成1-4个矩形
        num_rects = random.randint(self.min_masks, self.max_masks)
        
        for _ in range(num_rects):
            x1 = random.randint(0, self.img_size // 2)
            y1 = random.randint(0, self.img_size // 2)
            x2 = random.randint(x1 + 20, self.img_size)
            y2 = random.randint(y1 + 20, self.img_size)
            
            # 随机决定透明度
            if self.use_semi_transparent and random.random() < self.semi_transparent_prob:
                value = random.uniform(self.transparency_range[0], self.transparency_range[1])
            else:
                value = 1.0
            
            # 叠加（取最大值）
            mask[y1:y2, x1:x2] = np.maximum(mask[y1:y2, x1:x2], value)
        
        return torch.from_numpy(mask).unsqueeze(0)


@DATASET_REGISTRY.register()
class PairedImageDataset(Dataset):
    """配对图像数据集（用于超分）"""

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        self.gt_folder = opt.get('dataroot_gt', '')
        self.lq_folder = opt.get('dataroot_lq', '')
        self.filename_tmpl = opt.get('filename_tmpl', '{}')

        # 获取文件列表
        self.paths = self._get_paths()

        # 数据增强
        self.use_hflip = opt.get('use_hflip', False)
        self.use_rot = opt.get('use_rot', False)

        # 图像尺寸
        self.gt_size = opt.get('gt_size', 128)
        self.scale = opt.get('scale', 2)
        self.lq_size = self.gt_size // self.scale

        print(f"配对图像数据集加载完成: {len(self.paths)} 对图像")

    def _get_paths(self):
        """获取图像路径对"""
        paths = []

        if os.path.exists(self.gt_folder):
            for filename in os.listdir(self.gt_folder):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    gt_path = os.path.join(self.gt_folder, filename)

                    # 构建对应的LQ路径
                    name_wo_ext = os.path.splitext(filename)[0]
                    lq_filename = self.filename_tmpl.format(name_wo_ext) + os.path.splitext(filename)[1]
                    lq_path = os.path.join(self.lq_folder, lq_filename)

                    if os.path.exists(lq_path):
                        paths.append((gt_path, lq_path))

        return paths

    def __len__(self):
        return max(len(self.paths), 1)

    def __getitem__(self, index):
        try:
            if len(self.paths) > 0:
                gt_path, lq_path = self.paths[index % len(self.paths)]

                # 加载图像
                gt = Image.open(gt_path).convert('RGB')
                lq = Image.open(lq_path).convert('RGB')

                # 转换为tensor
                gt = transforms.ToTensor()(gt)
                lq = transforms.ToTensor()(lq)

                # 数据增强
                if self.use_hflip and random.random() < 0.5:
                    gt = torch.flip(gt, [2])
                    lq = torch.flip(lq, [2])

                if self.use_rot and random.random() < 0.5:
                    k = random.randint(1, 3)
                    gt = torch.rot90(gt, k, [1, 2])
                    lq = torch.rot90(lq, k, [1, 2])

            else:
                # 生成随机数据
                gt = torch.rand(3, self.gt_size, self.gt_size)
                lq = torch.rand(3, self.lq_size, self.lq_size)

            return {
                'gt': gt,
                'lq': lq,
                'gt_path': gt_path if len(self.paths) > 0 else 'generated',
            }

        except Exception as e:
            print(f"配对数据加载错误 (index: {index}): {e}")
            return {
                'gt': torch.rand(3, self.gt_size, self.gt_size),
                'lq': torch.rand(3, self.lq_size, self.lq_size),
                'gt_path': 'error_fallback',
            }


def build_combined_dataset(inpaint_opt, sr_opt):
    """构建联合数据集的工厂函数"""
    from .combined_dataset import CombinedDataset
    return CombinedDataset({
        'inpaint': inpaint_opt,
        'sr': sr_opt
    })

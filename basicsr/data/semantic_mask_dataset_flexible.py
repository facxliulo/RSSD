"""
灵活的语义掩码数据集 - 支持图片和掩码不一一对应
作者: 2025-12-31
用途: 图像修复训练，随机组合掩码
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import torchvision.transforms as transforms
import cv2


class FlexibleSemanticMaskDataset(Dataset):
    """
    灵活的语义掩码数据集
    - 图片和掩码数量可以不同
    - 训练时随机选择掩码进行组合
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        # 基础配置
        self.image_dir = opt.get('image_dir', '')
        self.label_dir = opt.get('label_dir', '')
        self.img_size = opt.get('img_size', 256)

        # 掩码组合配置
        self.min_masks = opt.get('min_masks', 1)
        self.max_masks = opt.get('max_masks', 3)
        self.use_self_mask_prob = opt.get('use_self_mask_prob', 0.0)  # 默认不使用自身掩码

        # 半透明配置
        self.use_semi_transparent = opt.get('use_semi_transparent', True)
        self.semi_transparent_prob = opt.get('semi_transparent_prob', 0.5)
        self.transparency_range = opt.get('transparency_range', [0.3, 0.7])

        # 空间变换配置
        self.use_spatial_transform = opt.get('use_spatial_transform', True)

        # 覆盖率限制
        self.max_coverage = opt.get('max_coverage', 0.6)

        # 加载数据
        self.image_paths = self._get_image_paths()
        self.label_paths = self._get_label_paths()

        # 数据增强
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
        ])

        print(f"\n{'=' * 80}")
        print(f"灵活语义掩码数据集初始化:")
        print(f"  图像目录: {self.image_dir}")
        print(f"  标签目录: {self.label_dir}")
        print(f"  图像数量: {len(self.image_paths)}")
        print(f"  掩码数量: {len(self.label_paths)}")
        print(f"  掩码组合: {self.min_masks}-{self.max_masks} 张")
        print(f"  半透明: {'启用' if self.use_semi_transparent else '禁用'} (概率: {self.semi_transparent_prob})")
        print(f"  空间变换: {'启用' if self.use_spatial_transform else '禁用'}")
        print(f"  最大覆盖率: {self.max_coverage * 100:.0f}%")
        print(f"{'=' * 80}\n")

    def _get_image_paths(self):
        """获取所有图像路径"""
        image_dir = Path(self.image_dir)
        if not image_dir.exists():
            print(f"警告: 图像目录不存在: {self.image_dir}")
            return []

        extensions = ['.png', '.jpg', '.jpeg', '.bmp']
        image_paths = []
        for ext in extensions:
            image_paths.extend(sorted(image_dir.glob(f'*{ext}')))

        return [str(p) for p in image_paths]

    def _get_label_paths(self):
        """获取所有标签路径"""
        label_dir = Path(self.label_dir)
        if not label_dir.exists():
            print(f"警告: 标签目录不存在: {self.label_dir}")
            return []

        extensions = ['.png', '.jpg', '.jpeg', '.bmp']
        label_paths = []
        for ext in extensions:
            label_paths.extend(sorted(label_dir.glob(f'*{ext}')))

        return [str(p) for p in label_paths]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        try:
            # 加载图像
            image_path = self.image_paths[index]
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)

            # 生成组合掩码
            composite_mask = self._generate_composite_mask(index)
            composite_mask = torch.from_numpy(composite_mask).unsqueeze(0)  # (1, H, W)

            return {
                'img': image,
                'mask': composite_mask,
                'img_path': image_path,
            }

        except Exception as e:
            print(f"加载数据出错 (index: {index}): {e}")
            # 返回随机数据作为fallback
            return {
                'img': torch.rand(3, self.img_size, self.img_size),
                'mask': torch.rand(1, self.img_size, self.img_size),
                'img_path': 'error_fallback',
            }

    def _generate_composite_mask(self, image_index):
        """
        生成组合掩码

        策略:
        1. 随机选择1-3张掩码
        2. 可能包含自身对应的掩码（概率可配置）
        3. 对每个掩码进行空间变换
        4. 随机决定透明度（完全破损 or 半透明）
        5. 取最大值叠加
        6. 控制总覆盖率 < 60%
        """
        H, W = self.img_size, self.img_size
        composite_mask = np.zeros((H, W), dtype=np.float32)

        # 确定要叠加的掩码数量
        num_masks = random.randint(self.min_masks, self.max_masks)

        # 决定是否包含自身掩码
        use_self_mask = (random.random() < self.use_self_mask_prob and
                         len(self.label_paths) > image_index)

        masks_to_combine = []

        # 添加自身掩码（如果需要）
        if use_self_mask:
            # 尝试找到对应的掩码
            image_name = Path(self.image_paths[image_index]).stem
            self_label = None

            for label_path in self.label_paths:
                if Path(label_path).stem == image_name:
                    self_label = label_path
                    break

            if self_label:
                masks_to_combine.append(self_label)
                num_masks -= 1  # 减少一个随机掩码

        # 添加随机掩码
        if len(self.label_paths) > 0:
            random_labels = random.sample(
                self.label_paths,
                min(num_masks, len(self.label_paths))
            )
            masks_to_combine.extend(random_labels)

        # 如果没有掩码，生成随机掩码
        if len(masks_to_combine) == 0:
            return self._generate_random_mask(H, W)

        # 叠加掩码
        for label_path in masks_to_combine:
            try:
                # 加载掩码
                label = Image.open(label_path).convert('L')
                label = label.resize((W, H))
                label = np.array(label).astype(np.float32) / 255.0

                # 二值化
                label = (label > 0.5).astype(np.float32)

                # 空间变换（如果启用）
                if self.use_spatial_transform:
                    label = self._apply_spatial_transform(label)

                # 决定透明度
                if self.use_semi_transparent and random.random() < self.semi_transparent_prob:
                    # 半透明（0.3-0.7）
                    alpha = random.uniform(*self.transparency_range)
                    mask_value = alpha
                else:
                    # 完全破损（1.0）
                    mask_value = 1.0

                # 叠加（取最大值）
                composite_mask = np.maximum(composite_mask, label * mask_value)

                # 检查覆盖率
                coverage = composite_mask.sum() / (H * W)
                if coverage > self.max_coverage:
                    break  # 停止添加

            except Exception as e:
                print(f"加载掩码出错: {label_path}, {e}")
                continue

        return composite_mask

    def _apply_spatial_transform(self, mask):
        """
        对掩码应用空间变换
        - 旋转 (0°, 90°, 180°, 270°)
        - 翻转 (水平、垂直)
        - 缩放 (0.5x - 1.5x)
        """
        H, W = mask.shape

        # 随机旋转
        if random.random() < 0.5:
            k = random.choice([1, 2, 3])  # 90°, 180°, 270°
            mask = np.rot90(mask, k)

        # 随机翻转
        if random.random() < 0.5:
            mask = np.fliplr(mask)  # 水平翻转
        if random.random() < 0.5:
            mask = np.flipud(mask)  # 垂直翻转

        # 随机缩放
        if random.random() < 0.3:
            scale = random.uniform(0.5, 1.5)
            new_h, new_w = int(H * scale), int(W * scale)
            mask_resized = cv2.resize(mask, (new_w, new_h))

            # 裁剪或填充回原始尺寸
            if scale > 1.0:
                # 缩放后更大，裁剪中心
                start_h = (new_h - H) // 2
                start_w = (new_w - W) // 2
                mask = mask_resized[start_h:start_h + H, start_w:start_w + W]
            else:
                # 缩放后更小，填充0
                result = np.zeros((H, W), dtype=np.float32)
                start_h = (H - new_h) // 2
                start_w = (W - new_w) // 2
                result[start_h:start_h + new_h, start_w:start_w + new_w] = mask_resized
                mask = result

        # 确保形状正确
        if mask.shape != (H, W):
            mask = cv2.resize(mask, (W, H))

        return mask

    def _generate_random_mask(self, H, W):
        """生成随机掩码（作为fallback）"""
        mask = np.zeros((H, W), dtype=np.float32)

        # 生成随机矩形掩码
        num_rects = random.randint(1, 3)
        for _ in range(num_rects):
            x1 = random.randint(0, W // 2)
            y1 = random.randint(0, H // 2)
            x2 = random.randint(x1 + 20, W)
            y2 = random.randint(y1 + 20, H)

            if self.use_semi_transparent and random.random() < self.semi_transparent_prob:
                value = random.uniform(*self.transparency_range)
            else:
                value = 1.0

            mask[y1:y2, x1:x2] = np.maximum(mask[y1:y2, x1:x2], value)

        return mask


# ========== 测试代码 ==========
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--label_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='test_samples')
    parser.add_argument('--num_samples', type=int, default=5)
    args = parser.parse_args()

    # 创建数据集
    opt = {
        'image_dir': args.image_dir,
        'label_dir': args.label_dir,
        'img_size': 256,
        'min_masks': 1,
        'max_masks': 3,
        'use_self_mask_prob': 0.0,  # 不使用自身掩码
        'use_semi_transparent': True,
        'semi_transparent_prob': 0.5,
        'transparency_range': [0.3, 0.7],
        'use_spatial_transform': True,
        'max_coverage': 0.6,
    }

    dataset = FlexibleSemanticMaskDataset(opt)

    print(f"\n测试数据集，生成 {args.num_samples} 个样本...")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 生成样本
    for i in range(min(args.num_samples, len(dataset))):
        sample = dataset[i]

        image = sample['img']
        mask = sample['mask']

        # 转换为可保存的格式
        image_np = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        mask_np = (mask[0].numpy() * 255).astype(np.uint8)

        # 创建可视化
        masked_image = image_np.copy()
        masked_image[mask_np > 128] = [255, 0, 0]  # 红色标记掩码区域

        # 保存
        Image.fromarray(image_np).save(f'{args.output_dir}/sample_{i:03d}_image.png')
        Image.fromarray(mask_np).save(f'{args.output_dir}/sample_{i:03d}_mask.png')
        Image.fromarray(masked_image).save(f'{args.output_dir}/sample_{i:03d}_masked.png')

        coverage = mask_np.sum() / (mask_np.shape[0] * mask_np.shape[1] * 255)
        print(f"样本 {i}: 掩码覆盖率 {coverage * 100:.1f}%")

    print(f"\n✓ 测试完成！样本已保存到: {args.output_dir}")
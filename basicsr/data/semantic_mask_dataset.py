#!/usr/bin/env python3
"""
语义掩码增强数据集
使用真实的语义分割标注作为掩码，进行组合和增强
"""

import os
import cv2
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path

class SemanticMaskInpaintingDataset(Dataset):
    """
    语义掩码修复数据集
    
    特点:
    1. 使用真实语义分割掩码
    2. 组合多个掩码（自身+随机）
    3. 半透明掩码（模拟不同破损程度）
    4. 空间变换（旋转、缩放、平移）
    """
    
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        
        # 路径配置
        self.image_dir = opt.get('image_dir', '')
        self.label_dir = opt.get('label_dir', '')
        
        # 加载图像和标签对
        self.image_paths, self.label_paths = self._load_pairs()
        
        # 参数
        self.img_size = opt.get('img_size', 256)
        self.min_masks = opt.get('min_masks', 1)  # 最少掩码数
        self.max_masks = opt.get('max_masks', 3)  # 最多掩码数
        self.use_self_mask_prob = opt.get('use_self_mask_prob', 0.7)  # 使用自身掩码概率
        self.max_coverage = opt.get('max_coverage', 0.6)  # 最大覆盖率
        
        # 透明度配置
        self.use_semi_transparent = opt.get('use_semi_transparent', True)
        self.semi_transparent_prob = opt.get('semi_transparent_prob', 0.5)
        self.transparency_range = opt.get('transparency_range', [0.3, 0.7])
        self.full_damage_ratio = opt.get('full_damage_ratio', 0.3)  # 30%完全破损
        
        # 空间变换
        self.use_spatial_transform = opt.get('use_spatial_transform', True)
        
        print(f"语义掩码数据集初始化:")
        print(f"  图像数量: {len(self.image_paths)}")
        print(f"  掩码数量: {len(self.label_paths)}")
        print(f"  掩码组合: {self.min_masks}-{self.max_masks} 张")
        print(f"  使用自身掩码概率: {self.use_self_mask_prob}")
        print(f"  半透明概率: {self.semi_transparent_prob}")
        print(f"  空间变换: {self.use_spatial_transform}")
    
    def _load_pairs(self):
        """加载图像-标签对"""
        image_paths = []
        label_paths = []
        
        # 假设文件名对应
        image_files = sorted(Path(self.image_dir).glob('*.png')) + \
                     sorted(Path(self.image_dir).glob('*.jpg'))
        
        for img_file in image_files:
            # 查找对应的label
            label_file = Path(self.label_dir) / (img_file.stem + '.png')
            if not label_file.exists():
                label_file = Path(self.label_dir) / (img_file.stem + '.jpg')
            
            if label_file.exists():
                image_paths.append(str(img_file))
                label_paths.append(str(label_file))
        
        return image_paths, label_paths
    
    def _load_mask(self, mask_path, size):
        """加载并处理掩码"""
        mask = Image.open(mask_path).convert('L')
        mask = mask.resize(size)
        mask = np.array(mask).astype(np.float32)
        
        # 二值化（假设非零区域是物体）
        mask = (mask > 0).astype(np.float32)
        
        return mask
    
    def _spatial_transform_mask(self, mask):
        """对掩码进行空间变换"""
        h, w = mask.shape
        
        # 随机旋转
        if random.random() < 0.5:
            angle = random.choice([0, 90, 180, 270])
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            mask = cv2.warpAffine(mask, M, (w, h))
        
        # 随机翻转
        if random.random() < 0.5:
            mask = cv2.flip(mask, random.choice([0, 1, -1]))
        
        # 随机缩放和平移
        if random.random() < 0.5:
            scale = random.uniform(0.5, 1.5)
            tx = random.randint(-w//4, w//4)
            ty = random.randint(-h//4, h//4)
            
            M = np.float32([
                [scale, 0, tx],
                [0, scale, ty]
            ])
            mask = cv2.warpAffine(mask, M, (w, h))
        
        return mask
    
    def _generate_composite_mask(self, index):
        """
        生成组合掩码
        
        策略:
        1. 70%概率使用自身掩码
        2. 随机添加1-2个其他掩码
        3. 对每个掩码应用空间变换
        4. 随机透明度（30%完全破损，70%半透明）
        5. 控制总覆盖率 < 60%
        """
        h, w = self.img_size, self.img_size
        composite_mask = np.zeros((h, w), dtype=np.float32)
        
        # 确定掩码数量
        num_masks = random.randint(self.min_masks, self.max_masks)
        
        masks_to_combine = []
        
        # 1. 自身掩码
        if random.random() < self.use_self_mask_prob:
            self_mask = self._load_mask(self.label_paths[index], (w, h))
            masks_to_combine.append(('self', self_mask))
        
        # 2. 随机其他掩码
        num_random = num_masks - len(masks_to_combine)
        if num_random > 0:
            random_indices = random.sample(
                range(len(self.label_paths)), 
                min(num_random, len(self.label_paths))
            )
            for idx in random_indices:
                if idx != index:  # 避免重复
                    random_mask = self._load_mask(self.label_paths[idx], (w, h))
                    masks_to_combine.append(('random', random_mask))
        
        # 3. 组合掩码
        for mask_type, mask in masks_to_combine:
            # 空间变换（对随机掩码）
            if mask_type == 'random' and self.use_spatial_transform:
                mask = self._spatial_transform_mask(mask)
            
            # 确定透明度
            if self.use_semi_transparent and random.random() < self.semi_transparent_prob:
                # 半透明
                if random.random() < self.full_damage_ratio:
                    # 30%区域完全破损
                    value = 1.0
                else:
                    # 70%区域半透明
                    value = random.uniform(*self.transparency_range)
            else:
                # 完全破损
                value = 1.0
            
            # 叠加（取最大值）
            composite_mask = np.maximum(composite_mask, mask * value)
            
            # 检查覆盖率
            current_coverage = composite_mask.mean()
            if current_coverage > self.max_coverage:
                # 如果覆盖率过高，停止添加
                break
        
        # 最终覆盖率控制
        if composite_mask.mean() > self.max_coverage:
            # 随机移除部分区域
            scale = self.max_coverage / composite_mask.mean()
            threshold = 1.0 - scale
            random_mask = np.random.rand(h, w)
            composite_mask[random_mask < threshold] = 0
        
        return composite_mask
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        try:
            # 1. 加载图像
            img_path = self.image_paths[index]
            image = Image.open(img_path).convert('RGB')
            image = image.resize((self.img_size, self.img_size))
            image_np = np.array(image).astype(np.float32) / 255.0
            
            # 2. 生成组合掩码
            composite_mask = self._generate_composite_mask(index)
            
            # 3. 转换为tensor
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float()
            mask_tensor = torch.from_numpy(composite_mask).unsqueeze(0).float()
            
            return {
                'img': image_tensor,           # 输入图像
                'mask': mask_tensor,           # 组合掩码
                'gt': image_tensor.clone(),    # Ground truth
                'img_path': img_path,
                'coverage': composite_mask.mean(),  # 覆盖率
            }
            
        except Exception as e:
            print(f"数据加载错误 (index: {index}): {e}")
            # 返回随机数据
            return {
                'img': torch.rand(3, self.img_size, self.img_size),
                'mask': torch.rand(1, self.img_size, self.img_size),
                'gt': torch.rand(3, self.img_size, self.img_size),
                'img_path': 'error',
                'coverage': 0.5,
            }


# 注册数据集
try:
    from basicsr.utils.registry import DATASET_REGISTRY
    
    @DATASET_REGISTRY.register()
    class SemanticMaskInpaintingDatasetRegistered(SemanticMaskInpaintingDataset):
        pass
    
except ImportError:
    print("Warning: 无法注册数据集")


def create_data_lists(data_root):
    """
    创建数据列表
    
    Args:
        data_root: ../../../Datasets/MuralDH/Mural_seg
    """
    train_images = Path(data_root) / 'train' / 'images'
    train_labels = Path(data_root) / 'train' / 'labels'
    test_images = Path(data_root) / 'test' / 'images'
    test_labels = Path(data_root) / 'test' / 'labels'
    
    # 训练集
    if train_images.exists():
        with open('train_images.flist', 'w') as f:
            for img in sorted(train_images.glob('*.png')):
                f.write(str(img.absolute()) + '\n')
        print(f"✓ train_images.flist")
    
    if train_labels.exists():
        with open('train_labels.flist', 'w') as f:
            for label in sorted(train_labels.glob('*.png')):
                f.write(str(label.absolute()) + '\n')
        print(f"✓ train_labels.flist")
    
    # 测试集
    if test_images.exists():
        with open('test_images.flist', 'w') as f:
            for img in sorted(test_images.glob('*.png')):
                f.write(str(img.absolute()) + '\n')
        print(f"✓ test_images.flist")
    
    if test_labels.exists():
        with open('test_labels.flist', 'w') as f:
            for label in sorted(test_labels.glob('*.png')):
                f.write(str(label.absolute()) + '\n')
        print(f"✓ test_labels.flist")


if __name__ == '__main__':
    # 示例用法
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='../../../Datasets/MuralDH/Mural_seg')
    args = parser.parse_args()
    
    print("创建数据列表...")
    create_data_lists(args.data_root)
    
    print("\n测试数据集...")
    opt = {
        'image_dir': str(Path(args.data_root) / 'train' / 'images'),
        'label_dir': str(Path(args.data_root) / 'train' / 'labels'),
        'img_size': 256,
        'min_masks': 1,
        'max_masks': 3,
        'use_self_mask_prob': 0.7,
        'use_semi_transparent': True,
        'semi_transparent_prob': 0.5,
    }
    
    dataset = SemanticMaskInpaintingDataset(opt)
    print(f"\n数据集大小: {len(dataset)}")
    
    # 测试第一个样本
    sample = dataset[0]
    print(f"图像形状: {sample['img'].shape}")
    print(f"掩码形状: {sample['mask'].shape}")
    print(f"覆盖率: {sample['coverage']:.2%}")

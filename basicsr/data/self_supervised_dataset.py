#!/usr/bin/env python3
"""
自监督图像修复数据集
适用于只有破损图片的情况
"""

import os
import cv2
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class SelfSupervisedInpaintingDataset(Dataset):
    """
    自监督修复数据集
    
    原理:
    1. 输入破损图片
    2. 检测真实破损区域 (damage_mask)
    3. 在未破损区域随机加掩码 (random_mask)
    4. 用未破损原图作为GT
    5. 训练修复random_mask，测试时修复damage_mask
    """
    
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        
        # 图像列表
        self.image_flist = opt.get('image_flist', '')
        self.image_paths = self._load_flist(self.image_flist)
        
        # 破损掩码列表（如果有）
        self.damage_mask_flist = opt.get('damage_mask_flist', '')
        if self.damage_mask_flist:
            self.damage_mask_paths = self._load_flist(self.damage_mask_flist)
        else:
            self.damage_mask_paths = None
        
        # 随机掩码列表（用于训练）
        self.random_mask_flist = opt.get('random_mask_flist', '')
        self.random_mask_paths = self._load_flist(self.random_mask_flist)
        
        # 参数
        self.img_size = opt.get('img_size', 256)
        self.damage_threshold = opt.get('damage_threshold', 0.3)  # 破损比例阈值
        self.use_detected_mask = opt.get('use_detected_mask', True)
        
        # 在线检测破损
        self.detect_damage_online = opt.get('detect_damage_online', False)
        
        print(f"自监督数据集初始化:")
        print(f"  图像数量: {len(self.image_paths)}")
        if self.damage_mask_paths:
            print(f"  破损掩码: {len(self.damage_mask_paths)}")
        else:
            print(f"  破损掩码: 在线检测")
        print(f"  随机掩码: {len(self.random_mask_paths)}")
    
    def _load_flist(self, flist_path):
        """加载文件列表"""
        if not flist_path or not os.path.exists(flist_path):
            return []
        
        paths = []
        with open(flist_path, 'r') as f:
            for line in f:
                path = line.strip()
                if path and os.path.exists(path):
                    paths.append(path)
        return paths
    
    def _detect_damage(self, image_np):
        """
        在线检测破损区域
        使用简单的自适应阈值
        """
        # 转灰度
        if len(image_np.shape) == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_np
        
        # 自适应阈值
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11, 2
        )
        
        # 形态学清理
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        return thresh.astype(np.float32) / 255.0
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        try:
            # 1. 加载破损图像
            img_path = self.image_paths[index]
            image = Image.open(img_path).convert('RGB')
            image = image.resize((self.img_size, self.img_size))
            image_np = np.array(image).astype(np.float32) / 255.0
            
            # 2. 获取或检测破损掩码
            if self.damage_mask_paths and index < len(self.damage_mask_paths):
                # 使用预生成的破损掩码
                damage_mask_path = self.damage_mask_paths[index]
                damage_mask = Image.open(damage_mask_path).convert('L')
                damage_mask = damage_mask.resize((self.img_size, self.img_size))
                damage_mask = np.array(damage_mask).astype(np.float32) / 255.0
                damage_mask = (damage_mask > 0.5).astype(np.float32)
            elif self.detect_damage_online:
                # 在线检测破损
                damage_mask = self._detect_damage((image_np * 255).astype(np.uint8))
            else:
                # 无破损掩码，假设全图可用
                damage_mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)
            
            # 3. 计算未破损区域
            undamaged_ratio = 1.0 - damage_mask.mean()
            
            # 如果破损太多，跳过这张图
            if undamaged_ratio < (1.0 - self.damage_threshold):
                # 破损区域超过阈值，使用随机数据
                image_np = np.random.rand(self.img_size, self.img_size, 3).astype(np.float32)
                damage_mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)
            
            # 4. 在未破损区域加随机掩码
            random_mask_idx = random.randint(0, len(self.random_mask_paths) - 1)
            random_mask_path = self.random_mask_paths[random_mask_idx]
            random_mask = Image.open(random_mask_path).convert('L')
            random_mask = random_mask.resize((self.img_size, self.img_size))
            random_mask = np.array(random_mask).astype(np.float32) / 255.0
            random_mask = (random_mask > 0.5).astype(np.float32)
            
            # 5. 确保随机掩码不覆盖破损区域
            # random_mask = random_mask * (1.0 - damage_mask)
            # 实际上可以覆盖，因为我们只在未破损区域计算损失
            
            # 6. 生成训练样本
            # GT: 未破损的原图
            gt = image_np.copy()
            
            # Input: 加了随机掩码的图
            masked_image = image_np * (1.0 - random_mask[..., np.newaxis])
            
            # 7. 转换为tensor
            gt_tensor = torch.from_numpy(gt).permute(2, 0, 1).float()
            masked_tensor = torch.from_numpy(masked_image).permute(2, 0, 1).float()
            random_mask_tensor = torch.from_numpy(random_mask).unsqueeze(0).float()
            damage_mask_tensor = torch.from_numpy(damage_mask).unsqueeze(0).float()
            
            # 8. 创建损失掩码（只在未破损区域计算损失）
            loss_mask = (1.0 - damage_mask_tensor)
            
            return {
                'gt': gt_tensor,                      # Ground truth (原图)
                'img': masked_tensor,                 # 输入（加掩码）
                'mask': random_mask_tensor,           # 训练用掩码
                'damage_mask': damage_mask_tensor,    # 真实破损掩码
                'loss_mask': loss_mask,               # 损失计算掩码
                'img_path': img_path,
                'undamaged_ratio': undamaged_ratio,   # 未破损比例
            }
            
        except Exception as e:
            print(f"数据加载错误 (index: {index}): {e}")
            # 返回随机数据
            return {
                'gt': torch.rand(3, self.img_size, self.img_size),
                'img': torch.rand(3, self.img_size, self.img_size),
                'mask': torch.rand(1, self.img_size, self.img_size),
                'damage_mask': torch.zeros(1, self.img_size, self.img_size),
                'loss_mask': torch.ones(1, self.img_size, self.img_size),
                'img_path': 'error',
                'undamaged_ratio': 1.0,
            }


class HybridInpaintingDataset(Dataset):
    """
    混合数据集：干净图片 + 破损图片
    
    训练时随机选择:
    - 70% 干净图片 + 随机掩码（标准训练）
    - 30% 破损图片 + 自监督训练
    """
    
    def __init__(self, clean_opt, damaged_opt, clean_ratio=0.7):
        super().__init__()
        
        self.clean_dataset = None
        self.damaged_dataset = None
        
        # 干净图片数据集
        if clean_opt:
            from basicsr.data.inpainting_dataset import InpaintingDataset
            self.clean_dataset = InpaintingDataset(clean_opt)
        
        # 破损图片数据集
        if damaged_opt:
            self.damaged_dataset = SelfSupervisedInpaintingDataset(damaged_opt)
        
        self.clean_ratio = clean_ratio
        
        # 计算总长度
        clean_len = len(self.clean_dataset) if self.clean_dataset else 0
        damaged_len = len(self.damaged_dataset) if self.damaged_dataset else 0
        self.total_len = clean_len + damaged_len
        
        print(f"混合数据集:")
        print(f"  干净图片: {clean_len}")
        print(f"  破损图片: {damaged_len}")
        print(f"  总计: {self.total_len}")
        print(f"  干净比例: {clean_ratio}")
    
    def __len__(self):
        return self.total_len
    
    def __getitem__(self, index):
        # 随机选择数据源
        if random.random() < self.clean_ratio and self.clean_dataset:
            # 使用干净数据
            idx = index % len(self.clean_dataset)
            return self.clean_dataset[idx]
        elif self.damaged_dataset:
            # 使用破损数据
            idx = index % len(self.damaged_dataset)
            return self.damaged_dataset[idx]
        else:
            # 默认返回随机数据
            return {
                'gt': torch.rand(3, 256, 256),
                'img': torch.rand(3, 256, 256),
                'mask': torch.rand(1, 256, 256),
                'img_path': 'fallback',
            }


# 注册数据集
try:
    from basicsr.utils.registry import DATASET_REGISTRY
    
    @DATASET_REGISTRY.register()
    class SelfSupervisedInpaintingDatasetRegistered(SelfSupervisedInpaintingDataset):
        pass
    
    @DATASET_REGISTRY.register()
    class HybridInpaintingDatasetRegistered(HybridInpaintingDataset):
        pass
    
except ImportError:
    print("Warning: 无法注册数据集到DATASET_REGISTRY")

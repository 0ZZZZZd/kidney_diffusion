import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
import random
import numpy as np


class KidneyDataset(Dataset):
    """肾脏CT增强数据集 - 2-to-1版本

    输入: 平扫(NC) + 第一期增强(CME)
    输出: 第三期增强(NP)
    """

    def __init__(self, config, phase="train"):
        self.config = config
        self.phase = phase

        data_root = config["path"]["data_root"]

        # 三个期相的目录
        self.nonenh_dir = os.path.join(data_root, config["data"]["nonenh_dir"])    # 平扫 NC
        self.phase1_dir = os.path.join(data_root, config["data"]["phase1_dir"])    # 第一期 CME
        self.phase3_dir = os.path.join(data_root, config["data"]["phase3_dir"])    # 第三期 NP (gt)

        # 获取文件列表（以平扫为基准）
        self.files = sorted([
            f for f in os.listdir(self.nonenh_dir)
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ])

        # 验证三个期相文件一致性
        phase1_files = sorted([f for f in os.listdir(self.phase1_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        phase3_files = sorted([f for f in os.listdir(self.phase3_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

        assert len(self.files) == len(phase1_files) == len(phase3_files),             f"三个期相文件数量不一致: nonenh={len(self.files)}, phase1={len(phase1_files)}, phase3={len(phase3_files)}"

        # 检查文件名匹配（去掉扩展名后应该一致）
        nonenh_names = [f.split('.')[0] for f in self.files]
        phase1_names = [f.split('.')[0] for f in phase1_files]
        phase3_names = [f.split('.')[0] for f in phase3_files]

        assert nonenh_names == phase1_names == phase3_names,             "三个期相的文件名不匹配，请确保文件名一致"

        # 构建完整路径
        self.nonenh_paths = [os.path.join(self.nonenh_dir, f) for f in self.files]
        self.phase1_paths = [os.path.join(self.phase1_dir, f) for f in self.files]
        self.phase3_paths = [os.path.join(self.phase3_dir, f) for f in self.files]

        # 数据增强
        if phase == "train" and config["data"].get("augmentation"):
            self.transform = self._get_transform_with_augmentation()
        else:
            self.transform = self._get_transform()

    def _get_transform(self):
        """基础预处理（无增强）"""
        return transforms.Compose([
            transforms.Resize((self.config["model"]["image_size"],
                               self.config["model"]["image_size"])),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # 归一化到[-1,1]
        ])

    def _get_transform_with_augmentation(self):
        """训练时数据增强"""
        aug_config = self.config["data"].get("augmentation", {})
        return transforms.Compose([
            transforms.Resize((self.config["model"]["image_size"],
                               self.config["model"]["image_size"])),
            transforms.RandomHorizontalFlip(p=aug_config.get("horizontal_flip", 0)),
            transforms.RandomRotation(aug_config.get("rotation", 0)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # 加载条件图像1：平扫 (NC)
        nonenh = Image.open(self.nonenh_paths[idx]).convert('L')
        nonenh = self.transform(nonenh)

        # 加载条件图像2：第一期增强 (CME)
        phase1 = Image.open(self.phase1_paths[idx]).convert('L')
        phase1 = self.transform(phase1)

        # 加载目标图像：第三期增强 (NP)
        phase3 = Image.open(self.phase3_paths[idx]).convert('L')
        phase3 = self.transform(phase3)

        # 将两个条件图像拼接 [2, H, W]
        cond_image = torch.cat([nonenh, phase1], dim=0)  # [2, H, W]

        return {
            "cond_image": cond_image,      # [2, H, W] - 平扫+第一期
            "gt_image": phase3,            # [1, H, W] - 第三期
            "path": self.files[idx],       # 文件名
            "nonenh": nonenh,              # [1, H, W] - 单独保留平扫（可选）
            "phase1": phase1                 # [1, H, W] - 单独保留第一期（可选）
        }

    @staticmethod
    def collate_fn(batch):
        """自定义collate函数"""
        return {
            "cond_image": torch.stack([item["cond_image"] for item in batch]),  # [B, 2, H, W]
            "gt_image": torch.stack([item["gt_image"] for item in batch]),    # [B, 1, H, W]
            "path": [item["path"] for item in batch],
            "nonenh": torch.stack([item["nonenh"] for item in batch]),        # [B, 1, H, W]
            "phase1": torch.stack([item["phase1"] for item in batch])         # [B, 1, H, W]
        }

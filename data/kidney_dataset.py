import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
import random
import numpy as np


class KidneyDataset(Dataset):
    """肾脏CT增强数据集"""

    def __init__(self, config, phase="train"):
        self.config = config
        self.phase = phase

        data_root = config["path"]["data_root"]
        nonenh_dir = os.path.join(data_root, config["data"]["nonenh_dir"])
        enh_dir = os.path.join(data_root, config["data"]["enh_dir"])

        # 获取文件列表
        self.nonenh_files = sorted([
            f for f in os.listdir(nonenh_dir)
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ])

        self.enh_files = sorted([
            f for f in os.listdir(enh_dir)
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ])

        # 验证文件一致性
        assert len(self.nonenh_files) == len(self.enh_files), \
            "nonenh和enh文件数量不一致"
        assert [f.split('.')[0] for f in self.nonenh_files] == [f.split('.')[0] for f in self.enh_files], \
            "文件名不匹配"

        self.nonenh_paths = [os.path.join(nonenh_dir, f) for f in self.nonenh_files]
        self.enh_paths = [os.path.join(enh_dir, f) for f in self.enh_files]

        # 数据增强
        if phase == "train" and config["data"]["augmentation"]:
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
        aug_config = self.config["data"]["augmentation"]
        return transforms.Compose([
            transforms.Resize((self.config["model"]["image_size"],
                               self.config["model"]["image_size"])),
            transforms.RandomHorizontalFlip(p=aug_config.get("horizontal_flip", 0)),
            transforms.RandomRotation(aug_config.get("rotation", 0)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.nonenh_files)

    def __getitem__(self, idx):
        # 加载条件图像（平扫）
        cond = Image.open(self.nonenh_paths[idx]).convert('L')
        cond = self.transform(cond)

        # 加载目标图像（增强）
        gt = Image.open(self.enh_paths[idx]).convert('L')
        gt = self.transform(gt)

        return {
            "cond_image": cond,
            "gt_image": gt,
            "path": self.nonenh_files[idx]  # 文件名
        }

    @staticmethod
    def collate_fn(batch):
        """自定义collate函数"""
        return {
            "cond_image": torch.stack([item["cond_image"] for item in batch]),
            "gt_image": torch.stack([item["gt_image"] for item in batch]),
            "path": [item["path"] for item in batch]
        }
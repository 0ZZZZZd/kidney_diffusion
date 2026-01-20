import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os
from .unet import UNet
import json
import torch.nn.functional as F


class CTDiffusion:
    """CT图像增强扩散模型"""

    def __init__(self, config, device=None):
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加载模型
        self.model = UNet(config).to(self.device)

        # 初始化超参数
        self.timesteps = config["model"]["timesteps"]
        self.beta_start = config["model"]["beta_schedule"]["beta_start"]
        self.beta_end = config["model"]["beta_schedule"]["beta_end"]
        self.image_size = config["model"]["image_size"]

        # 计算beta, alpha, alpha_bar
        self.betas = self._get_beta_schedule()
        self.alphas = 1.0 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0).to(self.device)
        self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1.0 - self.alphas_bar)

        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config["train"]["lr"],
            weight_decay=config["train"]["weight_decay"]
        )

        self.step = 0

    def _get_beta_schedule(self):
        """获取beta调度"""
        schedule = self.config["model"]["beta_schedule"]["schedule"]

        if schedule == "linear":
            return torch.linspace(
                self.beta_start, self.beta_end, self.timesteps
            ).to(self.device)
        elif schedule == "cosine":
            # 实现cosine schedule
            s = 0.008
            steps = self.timesteps + 1
            x = torch.linspace(0, self.timesteps, steps)
            alphas_bar = torch.cos(((x / self.timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
            alphas_bar = alphas_bar / alphas_bar[0]
            betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
            return torch.clip(betas, 0.001, 0.999)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

    def q_sample(self, x_0, t, noise=None):
        """前向扩散过程：向图像添加噪声"""
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha_bar_t = self.sqrt_alphas_bar[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_bar[t].view(-1, 1, 1, 1)

        return sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * noise, noise

    def p_sample(self, x_t, t, cond):
        """逆向去噪过程：单步去噪"""
        t_tensor = torch.tensor([t], device=self.device, dtype=torch.long)

        # 预测噪声
        with torch.no_grad():
            pred_noise = self.model(x_t, t_tensor, cond)

        # 获取当前时间步的alpha
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alphas_bar[t]

        # 计算beta
        beta_t = self.betas[t]

        # 计算去噪后的图像
        if t > 0:
            noise = torch.randn_like(x_t)
        else:
            noise = torch.zeros_like(x_t)

        alpha_bar_prev = self.alphas_bar[t - 1] if t > 0 else torch.tensor(1.0, device=self.device)
        sigma_t = torch.sqrt(beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar_t))

        # 去噪公式
        x_prev = (1 / torch.sqrt(alpha_t)) * (
                x_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * pred_noise
        ) + sigma_t * noise

        return x_prev

    @torch.no_grad()
    def sample(self, cond, n_samples=None, timesteps=None, save_intermediate=False):
        """完整采样过程"""
        self.model.eval()

        n_samples = n_samples or cond.shape[0]
        timesteps = timesteps or self.timesteps

        # 从纯噪声开始
        shape = (n_samples, 1, self.image_size, self.image_size)
        x = torch.randn(shape, device=self.device)

        # 复制条件图像
        if cond.shape[0] != n_samples:
            cond = cond[:n_samples]

        # 逐步去噪
        progress = tqdm(range(timesteps - 1, -1, -1), desc="Sampling")
        intermediate = [x] if save_intermediate else None

        for t in progress:
            x = self.p_sample(x, t, cond)
            if save_intermediate:
                intermediate.append(x.cpu())

        # 将x从[-1,1]转换到[0,1]
        x = (x + 1) / 2
        x = torch.clamp(x, 0, 1)

        return x, intermediate

    def train_step(self, data):
        """单步训练"""
        self.model.train()

        cond = data["cond_image"].to(self.device)  # [B, 1, H, W]
        x_0 = data["gt_image"].to(self.device)  # [B, 1, H, W]

        batch_size = x_0.shape[0]

        # 随机时间步
        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device)

        # 添加噪声
        noise = torch.randn_like(x_0)
        x_t, target_noise = self.q_sample(x_0, t, noise)

        # 预测噪声
        pred_noise = self.model(x_t, t, cond)

        # 计算损失
        loss = F.mse_loss(pred_noise, target_noise)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        if self.config["train"]["grad_clip"]:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config["train"]["grad_clip"]
            )

        self.optimizer.step()
        self.step += 1

        return loss.item()

    def save_checkpoint(self, path, epoch, loss):
        """保存模型"""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "step": self.step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": loss,
            "config": self.config
        }

        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.step = checkpoint["step"]
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
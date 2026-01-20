import torch
import torch.nn.functional as F
import numpy as np


def calculate_psnr(pred, target, max_val=1.0):
    """计算PSNR"""
    mse = torch.mean((pred - target) ** 2, dim=[1, 2, 3])
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr.mean().item()


def calculate_ssim(pred, target, window_size=11):
    """计算SSIM"""

    def gaussian_window(window_size, sigma=1.5):
        gauss = torch.tensor([
            np.exp(-(x - window_size // 2) ** 2 / (2 * sigma ** 2))
            for x in range(window_size)
        ])
        return gauss / gauss.sum()

    # 计算一维高斯核
    window_1d = gaussian_window(window_size).to(pred.device)
    window_2d = window_1d[:, None] * window_1d[None, :]
    window = window_2d.expand(1, 1, window_size, window_size)

    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=1)
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=1)

    mu1_sq, mu2_sq = mu1 ** 2, mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred ** 2, window, padding=window_size // 2, groups=1) - mu1_sq
    sigma2_sq = F.conv2d(target ** 2, window, padding=window_size // 2, groups=1) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size // 2, groups=1) - mu1_mu2

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
            (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )

    return ssim_map.mean().item()
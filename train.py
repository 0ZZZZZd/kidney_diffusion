import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
import os
from tqdm import tqdm
from models.ct_diffusion import CTDiffusion
from data.kidney_dataset import KidneyDataset
from utils.metrics import calculate_psnr, calculate_ssim
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-r", "--resume", default=None)
    return parser.parse_args()


def setup_logging(config):
    """创建日志目录"""
    os.makedirs(config["path"]["log_dir"], exist_ok=True)
    os.makedirs(config["path"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(config["path"]["sample_dir"], exist_ok=True)


def train_epoch(model, dataloader, epoch, config):
    """训练一个epoch"""
    total_loss = 0
    progress = tqdm(dataloader, desc=f"Epoch {epoch}")

    for i, data in enumerate(progress):
        loss = model.train_step(data)
        total_loss += loss

        # 更新进度条
        progress.set_postfix({"loss": f"{loss:.6f}"})

        # 日志记录
        if i % config["train"]["log_interval"] == 0:
            with open(os.path.join(config["path"]["log_dir"], "train.log"), "a") as f:
                f.write(f"Epoch {epoch}, iter {i}, loss: {loss:.6f}\n")

    return total_loss / len(dataloader)


def validate(model, val_dataloader, epoch, sample_dir):
    """验证并生成样本"""
    model.model.eval()

    # 取第一个batch验证
    data = next(iter(val_dataloader))
    cond = data["cond_image"].to(model.device)
    gt = data["gt_image"].to(model.device)

    # 生成样本
    with torch.no_grad():
        samples, _ = model.sample(cond, n_samples=4)

    # 保存样本
    from torchvision.utils import save_image
    save_image(samples, os.path.join(sample_dir, f"epoch_{epoch}.png"), nrow=2)

    # 计算指标
    psnr = calculate_psnr(samples, gt[:4])
    ssim = calculate_ssim(samples, gt[:4])

    return psnr, ssim


def main():
    args = parse_args()

    # 加载配置
    with open(args.config, "r") as f:
        config = json.load(f)

    # 设置日志
    setup_logging(config)

    # 创建模型
    model = CTDiffusion(config)

    # 恢复训练
    if args.resume:
        model.load_checkpoint(args.resume)

    # 创建数据集
    train_dataset = KidneyDataset(config, phase="train")
    val_dataset = KidneyDataset(config, phase="test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        pin_memory=config["data"]["pin_memory"],
        collate_fn=KidneyDataset.collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=2,
        collate_fn=KidneyDataset.collate_fn
    )

    # 训练循环
    start_epoch = 0
    best_loss = float("inf")

    for epoch in range(start_epoch, config["train"]["epochs"]):
        # 训练
        avg_loss = train_epoch(model, train_loader, epoch, config)  # 把 config 传进去
        print(f"Epoch {epoch}: Average Loss = {avg_loss:.6f}")

        # 验证
        if epoch % config["train"]["val_interval"] == 0:
            psnr, ssim = validate(model, val_loader, epoch, config["path"]["sample_dir"])
            print(f"Validation PSNR: {psnr:.2f}, SSIM: {ssim:.4f}")

        # 保存checkpoint
        if epoch % config["train"]["save_interval"] == 0:
            checkpoint_path = os.path.join(
                config["path"]["checkpoint_dir"],
                f"epoch_{epoch}.pth"
            )
            model.save_checkpoint(checkpoint_path, epoch, avg_loss)

            # 保存最佳模型
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_path = os.path.join(
                    config["path"]["checkpoint_dir"],
                    "best.pth"
                )
                model.save_checkpoint(best_path, epoch, avg_loss)


if __name__ == "__main__":
    main()
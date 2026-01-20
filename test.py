import torch
import json
import os
import argparse
from models.ct_diffusion import CTDiffusion
from data.kidney_dataset import KidneyDataset
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-n", "--num_samples", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()

    # 加载配置
    with open(args.config, "r") as f:
        config = json.load(f)

    # 创建模型
    model = CTDiffusion(config)
    model.load_checkpoint(args.model)

    # 加载测试数据
    test_dataset = KidneyDataset(config, phase="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2
    )

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)

    # 生成结果
    model.model.eval()

    results = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            if i >= args.num_samples:
                break

            cond = data["cond_image"].to(model.device)
            gt = data["gt_image"].to(model.device)

            # 生成
            sample, _ = model.sample(cond)

            # 保存对比图
            comparison = torch.cat([
                (cond + 1) / 2,
                (gt + 1) / 2,
                sample
            ], dim=0)

            save_image(
                comparison,
                os.path.join(args.output, f"sample_{i:03d}.png"),
                nrow=3,
                normalize=False
            )

            # 保存单个结果
            save_image(
                sample,
                os.path.join(args.output, f"enhanced_{data['path'][0]}"),
                normalize=False
            )

    print(f"Saved {args.num_samples} samples to {args.output}")


if __name__ == "__main__":
    main()
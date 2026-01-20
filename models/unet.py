# models/unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.norm1   = nn.GroupNorm(8, in_channels)
        self.conv1   = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.norm2   = nn.GroupNorm(8, out_channels)
        self.conv2   = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.dropout = nn.Dropout(dropout)

        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        time_emb = F.silu(self.time_mlp(t))
        h = h + time_emb[:, :, None, None]

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(self.conv2(h))

        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv  = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = q.view(B, C, H * W).transpose(1, 2)
        k = k.view(B, C, H * W)
        v = v.view(B, C, H * W).transpose(1, 2)

        attn = torch.softmax(torch.bmm(q, k) / np.sqrt(C), dim=-1)
        out  = torch.bmm(attn, v).transpose(1, 2).view(B, C, H, W)

        return x + self.proj(out)


class UNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        in_channels      = config["model"]["in_channels"]      # 2
        out_channels     = config["model"]["out_channels"]     # 1
        base_channels    = config["model"]["base_channels"]    # 64
        channel_mult     = config["model"]["channel_mult"]     # [1,2,4,8]
        num_res_blocks   = config["model"]["num_res_blocks"]   # 2
        time_emb_dim     = base_channels * 4
        attention_res    = set(config["model"]["attention_resolutions"])  # {16}

        # 时间编码
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # 输入层
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # 下采样
        self.downs   = nn.ModuleList()
        ch           = base_channels
        current_size = config["model"]["image_size"]  # 256

        for i, mult in enumerate(channel_mult):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                layers = [
                    ResidualBlock(ch, out_ch, time_emb_dim),
                    AttentionBlock(out_ch) if current_size in attention_res else nn.Identity(),
                ]
                self.downs.append(nn.ModuleList(layers))
                ch = out_ch

            if i != len(channel_mult) - 1:
                self.downs.append(nn.ModuleList([nn.Conv2d(ch, ch, 3, stride=2, padding=1)]))
                current_size //= 2

        # 中间层
        self.mid = nn.ModuleList([
            ResidualBlock(ch, ch, time_emb_dim),
            AttentionBlock(ch),
            ResidualBlock(ch, ch, time_emb_dim),
        ])

        # 上采样
        self.ups   = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mult))):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks + 1):
                layers = [
                    ResidualBlock(ch, out_ch, time_emb_dim),
                    AttentionBlock(out_ch) if current_size in attention_res else nn.Identity(),
                ]
                self.ups.append(nn.ModuleList(layers))
                ch = out_ch

            if i != 0:
                self.ups.append(nn.ModuleList([nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1)]))
                current_size *= 2

        # 输出层
        self.norm_out = nn.GroupNorm(8, ch)
        self.conv_out = nn.Conv2d(ch, out_channels, 3, padding=1)

    def forward(self, x, t, cond):
        x = torch.cat([cond, x], dim=1)  # [B,2,H,W]
        t_emb = self.time_mlp(t)
        h     = self.conv_in(x)

        # 下采样并收集跳跃连接
        skips = []
        for layer_group in self.downs:
            if len(layer_group) == 2:  # ResBlock + Attention
                h = layer_group[0](h, t_emb)
                h = layer_group[1](h)
                skips.append(h)
            else:  # 下采样
                h = layer_group[0](h)

        # 中间层
        for layer in self.mid:
            if isinstance(layer, ResidualBlock):
                h = layer(h, t_emb)
            else:
                h = layer(h)

        # 上采样
        for layer_group in self.ups:
            if len(layer_group) == 2:  # ResBlock + Attention
                h = layer_group[0](h, t_emb)
                h = layer_group[1](h)
            else:  # 上采样
                h = layer_group[0](h)

        h = self.norm_out(h)
        h = F.silu(h)
        return self.conv_out(h)
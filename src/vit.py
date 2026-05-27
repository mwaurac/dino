from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        d_model: int,
        in_channels: int = 3,
    ):
        super().__init__()

        self.num_patches = (
            image_size // patch_size
        ) ** 2  # I am assuming the images are square

        self.proj = nn.Conv2d(
            in_channels, d_model, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).flatten(2).transpose(1, 2)

        return x  # (B, N, d_model)


class Attention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        qkv_bias: bool = False,
        attn_dropout: float = 0.0,
        proj_droupout: float = 0.0,
    ) -> None:
        super().__init__()

        self.n_heads = n_heads
        self.head_size = d_model // n_heads
        self.qkv = nn.Linear(d_model, n_heads * self.head_size * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(n_heads * self.head_size, d_model)
        self.proj_drop = nn.Dropout(proj_droupout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        ## self.qkv(x) does (B, N, C) -> (B, N, n_heads * head_size * 3)
        # reshape changes to (B, N, 3, n_heads, head_size)
        # permute changes to (3, B, n_heads, N, head_size)
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.n_heads, self.head_size)
            .permute(2, 0, 3, 1, 4)
        )

        q, k, v = qkv.unbind(0)

        if x.is_cuda:
            x = F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                is_causal=False,
            )
        else:
            attn = (q @ k.transpose(-2, -1)) * (self.head_size**-0.5)
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)

        return self.proj_drop(self.proj(x))


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        act_layer=nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()

        out_dim = out_dim or in_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        noise = (
            torch.empty(shape, dtype=x.dtype, device=x.device)
            .bernoulli_(keep)
            .div_(keep)
        )
        return x * noise


class Block(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = Attention(d_model, n_heads, qkv_bias, attn_dropout, drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, int(d_model * mlp_ratio), d_model, drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size: int,
        patch_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        in_channels: int = 3,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.patch_embed = PatchEmbedding(img_size, patch_size, d_model, in_channels)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_emb = nn.Parameter(torch.zeros(1, num_patches + 1, d_model))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    d_model,
                    n_heads,
                    mlp_ratio,
                    qkv_bias,
                    drop_rate,
                    attn_drop_rate,
                    dpr[i],
                )
                for i in range(n_layers)
            ]
        )

        self.norm = nn.LayerNorm(d_model)
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def interpolate_pos_encoding(self, x: torch.Tensor, w: int, h: int) -> torch.Tensor:
        npatch = x.shape[1] - 1
        N = self.pos_emb.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_emb
        class_pos = self.pos_emb[:, :1, :]
        patch_pos = self.pos_emb[:, 1:, :]
        dim = x.shape[-1]
        side = int(math.sqrt(N))
        patch_pos = patch_pos.reshape(1, side, side, dim).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(
            patch_pos,
            scale_factor=math.sqrt(npatch / N),
            mode="bicubic",
            align_corners=False,
        )
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, dim)
        return torch.cat((class_pos, patch_pos), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape

        x = self.patch_embed(x)  # B, N, d_model
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = x + self.interpolate_pos_encoding(x, W, H)
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x[:, 0]

    def get_last_selfattention(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = x + self.interpolate_pos_encoding(x, W, H)
        x = self.pos_drop(x)

        for i, block in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = block(x)
            else:
                # Return attention from the last block
                return block.attn(block.norm1(x))
        return x


class DINOHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        use_bn: bool = False,
        norm_last_layer: bool = True,
        nlayers: int = 3,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
    ):
        super().__init__()
        layers: List[nn.Module] = [nn.Linear(in_dim, hidden_dim)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        for _ in range(nlayers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, bottleneck_dim))
        self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad_(False)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        return self.last_layer(x)


def _build_vit_backbone(model_cfg: DictConfig) -> VisionTransformer:
    return VisionTransformer(
        img_size=224,
        patch_size=model_cfg.patch_size,
        d_model=model_cfg.d_model,
        n_layers=model_cfg.depth,
        n_heads=model_cfg.num_heads,
        mlp_ratio=model_cfg.mlp_ratio,
        qkv_bias=model_cfg.qkv_bias,
        drop_path_rate=model_cfg.drop_path_rate,
    )

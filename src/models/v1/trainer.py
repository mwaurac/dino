from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from src.augmentations import MultiCropDataset
from src.loss import DINOLoss
from src.models.v1 import build_student_teacher
from src.vit import DINOHead


class MultiCropWrapper(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module) -> None:
        super().__init__()

        self.backbone = backbone
        self.head = head

    def forward(self, crops: list[torch.Tensor]) -> torch.Tensor:
        # concat all crops along the batch dim for efficiency
        n_crops = len(crops)
        # crops: list of (B, C, H, W) may have difference H/W (global vs local)
        # Group by shape to minimize calls
        idx_crops = {}
        for idx, crop in enumerate(crops):
            key = tuple(crop.shape[-2:])
            idx_crops.setdefault(key, []).append(idx)

        outputs = [None] * n_crops
        for shape, idxs in idx_crops.items():
            x = torch.cat([crops[idx] for idx in idxs])
            y = self.head(self.backbone(x))
            chunk_size = y.shape[0] // len(idxs)
            for j, idx in enumerate(idxs):
                outputs[idx] = y[j * chunk_size:(j + 1) * chunk_size]

        return torch.cat(outputs)

def _collate_crops(batch: list) -> list[torch.Tensor]:
    n_crops = len(batch[0])
    return [torch.stack([b[i] for b in batch]) for i in range(n_crops)]

class DINOv1Trainer:
    def __init__(
            self,
            cfg: DictConfig,
            device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.device = device
        self.tcfg = cfg.pretrain
        self.mcfg = cfg.model


        self.student_backbone, self.teacher_backbone = build_student_teacher()
        self.student_head = DINOHead(
            in_dim=self.mcfg.d_model,
            out_dim=self.tcfg.out_dim,
            norm_last_layer=self.tcfg.norm_last_layer,
        )
        self.teacher_head = DINOHead(
            in_dim=self.mcfg.d_model,
            out_dim=self.tcfg.out_dim,
            norm_last_layer=self.tcfg.norm_last_layer,
        )

        for p in self.teacher_head.parameters():
            p.requires_grad_(False)

        self.student = MultiCropWrapper(self.student_backbone, self.student_head).to(device)
        self.teacher = MultiCropWrapper(self.teacher_backbone, self.teacher_head).to(device)
        # copy the student weights into teacher
        self.teacher.load_state_dict(self.student.state_dict())

        self.loss_fn = DINOLoss(
            out_dim=self.tcfg.out_dim,
            student_temp=self.tcfg.student_temp,
            teacher_temp=self.tcfg.teacher_temp,
            warmup_teacher_temp=self.tcfg.warmup_teacher_temp,
            warmup_teacher_temp_epochs=self.tcfg.warmup_teacher_temp_epochs,
            nepochs=self.tcfg.nepochs,
        ).to(device)

        # data
        base_ds = ImageFolder(self.tcfg.data_path)
        ds = MultiCropDataset(
            base_ds,
            global_crops_scale=tuple(self.tcfg.global_crops_scale),
            local_crops_scale=tuple(self.tcfg.local_crops_scale),
            local_crops_number=tuple(self.tcfg.local_crops_number),
            image_size=self.tcfg.image_size,
            local_crop_size=self.tcfg.local_crop_size,
        )
        self.loader = DataLoader(
            ds,
            batch_size=self.tcfg.batch_size,
            shuffle=True,
            num_workers=self.tcfg.num_workers,
            pin_memory=self.tcfg.pin_memory,
            drop_last=True,
            collate_fn=_collate_crops
        )

        # optimizer
        params = [p for p in self.teacher_head.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(
            params,
            lr=self._scaled_lr(),
            weight_decay=self.tcfg.weight_decay_start,
        )

        self.start_epoch = 0
        if self.tcfg.resume:
            self._resume(self.tcfg.resume)

        # logging
        self.wandb = None
        try:
            import wandb
            wandb.init(project=self.tcfg.wandb_project, config=dict(cfg=self.tcfg))
            self.wandb = wandb
        except ImportError:
            print("[warn] wandb is not installed - skipping W&B logging")

        Path(self.tcfg.output_dir).mkdir(parents=True, exist_ok=True)

    def train(self) -> None:
        from rich.console import Console
        console = Console()
        console.rule("[bold green]DINOv1 Pretraining")

        for epoch in range(self.start_epoch, self.tcfg.nepochs):
            self._set_lr_wd(epoch)
            ep_loss = self._train_epoch(epoch)
            console.print(f"Epoch {epoch + 1:>4}/{self.tcfg.epochs}  loss={ep_loss:.4f}")

            if self.wandb:
                self.wandb.log({"loss": ep_loss, "epoch": epoch})

            if (epoch + 1) % self.tcfg.save_freq == 0 or epoch == self.tcfg.epochs - 1:
                self._save_checkpoint(epoch)

    def _train_epoch(self, epoch: int) -> float:
        self.student.train()
        self.teacher.eval()
        total_loss = 0.0
        n_batches = len(self.loader)

        for step, crops in enumerate(tqdm(self.loader, desc=f"Epoch {epoch + 1}", leave=False)):
            crops = [c.to(self.device, non_blocking=True) for c in crops]

            # teacher sees only global crops
            with torch.no_grad():
                teacher_output = self.teacher(crops[:2])
            student_output = self.student(crops)

            loss = self.loss_fn(student_output, teacher_output, epoch)
            self.optimizer.zero_grad()
            loss.backward()

            # clip gradients
            if self.tcfg.clip_grad:
                nn.utils.clip_grad_norm_(self.student.parameters(), self.tcfg.clip_grad)

            # freeze last layer for first N epochs
            if epoch < self.tcfg.freeze_last_layer:
                for n, p in self.student.named_parameters():
                    if "last_layer" in n:
                        p.grad = None

            self.optimizer.step()
            self._update_teacher(epoch)

            total_loss += loss.item()

            if step % self.tcfg.log_freq == 0 and self.wandb:
                self.wandb.log({"step_loss": loss.item})

        return total_loss / n_batches

    @torch.no_grad()
    def _update_teacher(self, epoch: int) -> None:
        m = self._momentum_schedule(epoch)
        for ps, pt in zip(self.student.parameters(), self.teacher.parameters()):
            pt.data.mul_(m).add_((1 - m) * ps.data)

    def _momentum_schedule(self, epoch: int) -> float:
        return self.tcfg.momentum_teacher + (1.0 - self.tcfg.momentum_teacher) * (
                1 - math.cos(math.pi * epoch / self.tcfg.epochs)
        ) / 2

    def _scaled_lr(self) -> float:
        return self.tcfg.base_lr * self.tcfg.batch_size / 256

    def _set_lr_wd(self, epoch: int) -> None:
        """Cosine-decay LR and linearly increase weight decay."""
        if epoch < self.tcfg.warmup_epochs:
            lr = self._scaled_lr() * epoch / self.tcfg.warmup_epochs
        else:
            progress = (epoch - self.tcfg.warmup_epochs) / (self.tcfg.epochs - self.tcfg.warmup_epochs)
            lr = self.tcfg.min_lr + (self._scaled_lr() - self.tcfg.min_lr) * (
                    1 + math.cos(math.pi * progress)
            ) / 2

        wd_progress = epoch / self.tcfg.epochs
        wd = self.tcfg.weight_decay_start + (
                self.tcfg.weight_decay_end - self.tcfg.weight_decay_start
        ) * wd_progress

        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
            pg["weight_decay"] = wd

    def _save_checkpoint(self, epoch: int) -> None:
        path = Path(self.tcfg.output_dir) / f"checkpoint_ep{epoch + 1:04d}.pth"
        torch.save({
            "epoch": epoch,
            "student": self.student.state_dict(),
            "teacher": self.teacher.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "loss_fn": self.loss_fn.state_dict(),
        }, path)

    def _resume(self, ckpt_path: str) -> None:
        state = torch.load(ckpt_path, map_location=self.device)
        self.student.load_state_dict(state["student"])
        self.teacher.load_state_dict(state["teacher"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.loss_fn.load_state_dict(state["loss_fn"])
        self.start_epoch = state["epoch"] + 1
        print(f"Resumed from epoch {self.start_epoch}")


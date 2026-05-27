from __future__ import annotations

from typing import List, Tuple

import torch
from PIL import Image, ImageFilter
from torchvision import transforms
from torchvision.transforms import functional as TF


class GaussianBlur:
    def __init__(
        self, p: float, radius_min: float = 0.1, radius_max: float = 2.0
    ) -> None:
        self.p = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img: Image.Image) -> Image.Image:
        if torch.rand(1).item() < self.p:
            radius = self.radius_min + torch.rand(1).item() * (
                self.radius_max - self.radius_min
            )
            return img.filter(ImageFilter.GaussianBlur(radius=radius))
        return img


class Solarize:
    def __init__(self, p: float) -> None:
        self.p = p

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p:
            return TF.solarize(img, threshold=128)
        return img


def make_global_transform(
    global_crops_scale: Tuple[float, float],
    image_size: int = 224,
    is_second_global: bool = False,
) -> transforms.Compose:
    flip = transforms.RandomHorizontalFlip(p=0.5)
    color_jitter = transforms.RandomApply(
        [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
        p=0.8,
    )
    gray = transforms.RandomGrayscale(p=0.2)
    # First global crop: strong blur, no solarize
    # Second global crop: weak blur, solarize
    blur = GaussianBlur(p=1.0 if not is_second_global else 0.1)
    solar = Solarize(p=0.2 if is_second_global else 0.0)
    norm = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size,
                scale=global_crops_scale,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            flip,
            color_jitter,
            gray,
            blur,
            transforms.ToTensor(),
            solar,
            norm,
        ]
    )


def make_local_transform(
    local_crops_scale: Tuple[float, float],
    local_crop_size: int = 96,
) -> transforms.Compose:
    flip = transforms.RandomHorizontalFlip(p=0.5)
    color_jitter = transforms.RandomApply(
        [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
        p=0.8,
    )
    gray = transforms.RandomGrayscale(p=0.2)
    blur = GaussianBlur(p=0.5)
    norm = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                local_crop_size,
                scale=local_crops_scale,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            flip,
            color_jitter,
            gray,
            blur,
            transforms.ToTensor(),
            norm,
        ]
    )


class MultiCropDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_dataset: torch.utils.data.Dataset,
        global_crops_scale: Tuple[float, float],
        local_crops_scale: Tuple[float, float],
        local_crops_number: int,
        image_size: int = 224,
        local_crop_size: int = 96,
    ):
        self.dataset = base_dataset
        self.global1 = make_global_transform(global_crops_scale, image_size, False)
        self.global2 = make_global_transform(global_crops_scale, image_size, True)
        self.local = make_local_transform(local_crops_scale, local_crop_size)
        self.local_crops_number = local_crops_number

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        crops: List[torch.Tensor] = [self.global1(img), self.global2(img)]
        for _ in range(self.local_crops_number):
            crops.append(self.local(img))
        return crops

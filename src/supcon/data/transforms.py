from __future__ import annotations

from typing import Callable

from omegaconf import DictConfig
from torchvision import transforms


class TwoCropTransform:
    def __init__(self, base_transform: Callable):
        self.base_transform = base_transform

    def __call__(self, x):
        v1 = self.base_transform(x)
        v2 = self.base_transform(x)
        return v1, v2


def build_cifar10_train_transform(cfg: DictConfig) -> transforms.Compose:
    color_jitter = transforms.ColorJitter(
        brightness=cfg.color_jitter.brightness,
        contrast=cfg.color_jitter.contrast,
        saturation=cfg.color_jitter.saturation,
        hue=cfg.color_jitter.hue,
    )

    return transforms.Compose([
        transforms.RandomResizedCrop(cfg.crop_size, scale=(cfg.scale_min, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([color_jitter], p=cfg.color_jitter_p),
        transforms.RandomGrayscale(p=cfg.grayscale_p),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=cfg.blur_kernel, sigma=(0.1, 2.0))],
            p=cfg.blur_p,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.normalize.mean, std=cfg.normalize.std),
    ])


def build_cifar10_supervised_train_transform(cfg: DictConfig) -> transforms.Compose:
    # Standard supervised CIFAR-10 augmentation (paper-style baseline / linear eval).
    return transforms.Compose([
        transforms.RandomCrop(cfg.crop_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.normalize.mean, std=cfg.normalize.std),
    ])


def build_cifar10_eval_transform(cfg: DictConfig) -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.normalize.mean, std=cfg.normalize.std),
    ])

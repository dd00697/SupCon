from __future__ import annotations

from typing import Tuple

import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision.datasets import CIFAR10

from supcon.data.transforms import (
    TwoCropTransform,
    build_cifar10_eval_transform,
    build_cifar10_supervised_train_transform,
    build_cifar10_train_transform,
)
from supcon.utils.seed import seed_worker


def build_cifar10_loaders(
    cfg: DictConfig,
    seed: int,
    two_crop_train: bool,
) -> Tuple[DataLoader, DataLoader]:
    data_root = to_absolute_path(cfg.root)

    if two_crop_train:
        train_transform = TwoCropTransform(build_cifar10_train_transform(cfg))
    else:
        train_transform = build_cifar10_supervised_train_transform(cfg)

    test_transform = build_cifar10_eval_transform(cfg)

    train_ds = CIFAR10(root=data_root, train=True, download=True, transform=train_transform)
    test_ds = CIFAR10(root=data_root, train=False, download=True, transform=test_transform)

    if int(cfg.train_subset) > 0:
        train_size = min(int(cfg.train_subset), len(train_ds))
        train_ds = Subset(train_ds, indices=range(train_size))
    if int(cfg.test_subset) > 0:
        test_size = min(int(cfg.test_subset), len(test_ds))
        test_ds = Subset(test_ds, indices=range(test_size))

    generator = torch.Generator()
    generator.manual_seed(seed)

    persistent = cfg.num_workers > 0

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=persistent,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=persistent,
    )

    return train_loader, test_loader

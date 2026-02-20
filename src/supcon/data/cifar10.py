from __future__ import annotations

from typing import Optional, Tuple

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


def _build_loader(
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    drop_last: bool,
    seed: int,
) -> DataLoader:
    effective_drop_last = bool(drop_last and len(dataset) >= batch_size)
    generator = torch.Generator()
    generator.manual_seed(seed)
    persistent = num_workers > 0
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=effective_drop_last,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=persistent,
    )


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

    train_loader = _build_loader(
        dataset=train_ds,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
        pin_memory=bool(cfg.pin_memory),
        drop_last=True,
        seed=seed,
    )

    test_loader = _build_loader(
        dataset=test_ds,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        pin_memory=bool(cfg.pin_memory),
        drop_last=False,
        seed=seed + 1,
    )

    return train_loader, test_loader


def build_cifar10_supervised_loaders(
    cfg: DictConfig,
    seed: int,
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
    data_root = to_absolute_path(cfg.root)
    train_transform = build_cifar10_supervised_train_transform(cfg)
    eval_transform = build_cifar10_eval_transform(cfg)

    full_train_aug = CIFAR10(root=data_root, train=True, download=True, transform=train_transform)
    full_train_eval = CIFAR10(root=data_root, train=True, download=True, transform=eval_transform)
    test_ds = CIFAR10(root=data_root, train=False, download=True, transform=eval_transform)

    total_train = len(full_train_aug)
    max_train = total_train if int(cfg.train_subset) <= 0 else min(int(cfg.train_subset), total_train)
    candidate_indices = list(range(max_train))

    val_size = max(int(cfg.val_size), 0)
    if val_size >= len(candidate_indices):
        val_size = max(len(candidate_indices) - 1, 0)

    split_gen = torch.Generator()
    split_gen.manual_seed(seed)
    perm = torch.randperm(len(candidate_indices), generator=split_gen).tolist()
    shuffled_indices = [candidate_indices[i] for i in perm]

    if val_size > 0:
        val_indices = shuffled_indices[:val_size]
        train_indices = shuffled_indices[val_size:]
    else:
        val_indices = []
        train_indices = shuffled_indices

    train_ds = Subset(full_train_aug, train_indices)
    val_ds = Subset(full_train_eval, val_indices) if len(val_indices) > 0 else None

    if int(cfg.test_subset) > 0:
        test_size = min(int(cfg.test_subset), len(test_ds))
        test_ds = Subset(test_ds, indices=range(test_size))

    train_loader = _build_loader(
        dataset=train_ds,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
        pin_memory=bool(cfg.pin_memory),
        drop_last=True,
        seed=seed,
    )
    val_loader = None
    if val_ds is not None:
        val_loader = _build_loader(
            dataset=val_ds,
            batch_size=int(cfg.batch_size),
            shuffle=False,
            num_workers=int(cfg.num_workers),
            pin_memory=bool(cfg.pin_memory),
            drop_last=False,
            seed=seed + 1,
        )

    test_loader = _build_loader(
        dataset=test_ds,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        pin_memory=bool(cfg.pin_memory),
        drop_last=False,
        seed=seed + 2,
    )
    return train_loader, val_loader, test_loader

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torchvision import models


def build_encoder(cfg: DictConfig) -> Tuple[nn.Module, int]:
    if cfg.name == "resnet18":
        encoder = models.resnet18(weights=None)
    elif cfg.name == "resnet50":
        encoder = models.resnet50(weights=None)
    else:
        raise ValueError(f"Unsupported backbone: {cfg.name}")

    if cfg.cifar_stem:
        encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        encoder.maxpool = nn.Identity()

    feat_dim = encoder.fc.in_features
    encoder.fc = nn.Identity()
    return encoder, feat_dim


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, depth: int = 2):
        super().__init__()
        if depth == 1:
            layers = [nn.Linear(in_dim, out_dim)]
        elif depth == 2:
            layers = [
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, out_dim),
            ]
        else:
            raise ValueError(f"Projection head depth must be 1 or 2, got {depth}")

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, dim=1)


class LinearHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)
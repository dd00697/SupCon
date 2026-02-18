from __future__ import annotations

import math
import time

import torch
from torch.optim.lr_scheduler import LambdaLR

from supcon.utils.meters import AverageMeter


def build_cosine_scheduler(optimizer: torch.optim.Optimizer, cfg_optim) -> LambdaLR:
    total_epochs = int(cfg_optim.epochs)
    warmup_epochs = int(cfg_optim.warmup_epochs)
    base_lr = float(cfg_optim.lr)
    min_lr = float(cfg_optim.min_lr)
    min_ratio = min_lr / max(base_lr, 1e-12)

    def lr_lambda(epoch_index: int) -> float:
        # epoch_index is 0-indexed and corresponds to the current epoch.
        # Torch schedulers apply an initial step on construction, so lambda(0) must be epoch 1.
        if total_epochs <= 1:
            return 1.0

        if warmup_epochs > 0 and epoch_index < warmup_epochs:
            return float(epoch_index + 1) / float(warmup_epochs)

        remaining = total_epochs - warmup_epochs
        if remaining <= 1:
            return 1.0

        t = epoch_index - warmup_epochs  # 0..remaining-1
        t = min(max(t, 0), remaining - 1)
        progress = float(t) / float(remaining - 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def train_one_epoch_pretrain(
    encoder: torch.nn.Module,
    projector: torch.nn.Module,
    criterion: torch.nn.Module,
    train_loader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
    accum_steps: int,
    grad_clip: float,
    print_freq: int,
) -> dict:
    encoder.train()
    projector.train()

    loss_meter = AverageMeter()
    start = time.time()

    optimizer.zero_grad(set_to_none=True)

    for step, (views, labels) in enumerate(train_loader):
        v1, v2 = views
        v1 = v1.to(device, non_blocking=True)
        v2 = v2.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
            z1 = projector(encoder(v1))
            z2 = projector(encoder(v2))
            feats = torch.cat([z1, z2], dim=0)
            targets = torch.cat([labels, labels], dim=0)
            loss = criterion(feats, targets)

        scaled_loss = loss / accum_steps
        scaler.scale(scaled_loss).backward()

        should_step = (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader)
        if should_step:
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                params = list(encoder.parameters()) + list(projector.parameters())
                torch.nn.utils.clip_grad_norm_(params, grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        loss_meter.update(float(loss.item()), labels.size(0))

        if print_freq > 0 and ((step + 1) % print_freq == 0 or (step + 1) == len(train_loader)):
            print(f"[pretrain] step {step + 1}/{len(train_loader)} loss={loss_meter.avg:.4f}")

    return {"loss": loss_meter.avg, "time": time.time() - start}

from __future__ import annotations

import time

import torch
import torch.nn as nn

from supcon.utils.meters import AverageMeter, accuracy


def train_one_epoch_ce(
    encoder: torch.nn.Module,
    linear_head: torch.nn.Module,
    criterion: nn.Module,
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
    linear_head.train()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    start = time.time()

    optimizer.zero_grad(set_to_none=True)

    for step, (images, labels) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
            feats = encoder(images)
            logits = linear_head(feats)
            loss = criterion(logits, labels)

        scaled_loss = loss / accum_steps
        scaler.scale(scaled_loss).backward()

        should_step = (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader)
        if should_step:
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                params = list(encoder.parameters()) + list(linear_head.parameters())
                torch.nn.utils.clip_grad_norm_(params, grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        acc1 = accuracy(logits, labels, topk=(1,))[0]
        loss_meter.update(float(loss.item()), labels.size(0))
        acc_meter.update(float(acc1.item()), labels.size(0))

        if print_freq > 0 and ((step + 1) % print_freq == 0 or (step + 1) == len(train_loader)):
            print(
                f"[ce/train] step {step + 1}/{len(train_loader)} "
                f"loss={loss_meter.avg:.4f} acc1={acc_meter.avg:.2f}"
            )

    return {"loss": loss_meter.avg, "acc1": acc_meter.avg, "time": time.time() - start}


@torch.no_grad()
def evaluate_ce(
    encoder: torch.nn.Module,
    linear_head: torch.nn.Module,
    criterion: nn.Module,
    data_loader,
    device: torch.device,
    amp_enabled: bool,
) -> dict:
    encoder.eval()
    linear_head.eval()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for images, labels in data_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
            feats = encoder(images)
            logits = linear_head(feats)
            loss = criterion(logits, labels)

        acc1 = accuracy(logits, labels, topk=(1,))[0]
        loss_meter.update(float(loss.item()), labels.size(0))
        acc_meter.update(float(acc1.item()), labels.size(0))

    return {"loss": loss_meter.avg, "acc1": acc_meter.avg}

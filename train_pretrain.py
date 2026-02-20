from __future__ import annotations

from pathlib import Path

import hydra
import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from supcon.data import build_cifar10_loaders
from supcon.losses import SupConLoss
from supcon.models import ProjectionHead, build_encoder
from supcon.train.pretrain import build_cosine_scheduler, train_one_epoch_pretrain
from supcon.utils import (
    init_wandb,
    load_checkpoint,
    log_wandb,
    patch_hydra_argparse,
    save_checkpoint,
    save_resolved_config,
    seed_everything,
)

patch_hydra_argparse()


@hydra.main(config_path="configs", config_name="pretrain", version_base=None)
def main(cfg: DictConfig) -> None:
    run_pretrain(cfg)


def run_pretrain(cfg: DictConfig) -> None:
    seed_everything(int(cfg.seed), deterministic=bool(cfg.deterministic))

    run_dir = Path(HydraConfig.get().runtime.output_dir)
    save_resolved_config(cfg, run_dir / "config_resolved.yaml")

    run_display_name = f"{cfg.run_name}_{cfg.model.name}_seed{cfg.seed}_bs{cfg.data.batch_size}"
    run = init_wandb(cfg, run_name=run_display_name)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(cfg.amp and device.type == "cuda")

    print(f"Device: {device}")
    train_loader, _ = build_cifar10_loaders(cfg.data, seed=int(cfg.seed), two_crop_train=True)

    encoder, feat_dim = build_encoder(cfg.model)
    projector = ProjectionHead(
        in_dim=feat_dim,
        hidden_dim=int(cfg.model.projector.hidden_dim),
        out_dim=int(cfg.model.projector.out_dim),
        depth=int(cfg.model.projector.depth),
    )

    encoder = encoder.to(device)
    projector = projector.to(device)

    criterion = SupConLoss(temperature=float(cfg.method.temperature))
    optimizer = torch.optim.SGD(
        list(encoder.parameters()) + list(projector.parameters()),
        lr=float(cfg.optim.lr),
        momentum=float(cfg.optim.momentum),
        weight_decay=float(cfg.optim.weight_decay),
    )
    scheduler = build_cosine_scheduler(optimizer, cfg.optim)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_loss = float("inf")
    best_epoch = 0
    start_epoch = 1
    ckpt_dir = run_dir / "checkpoints"
    resume_path = None

    if bool(cfg.resume):
        if str(cfg.resume_ckpt):
            resume_path = Path(to_absolute_path(str(cfg.resume_ckpt)))
        else:
            candidate = ckpt_dir / "last.ckpt"
            if candidate.exists():
                resume_path = candidate

        if resume_path is not None and resume_path.exists():
            resume_ckpt = load_checkpoint(resume_path, map_location="cpu")
            encoder.load_state_dict(resume_ckpt["encoder_state"], strict=True)
            projector.load_state_dict(resume_ckpt["projector_state"], strict=True)
            optimizer.load_state_dict(resume_ckpt["optimizer_state"])
            scheduler.load_state_dict(resume_ckpt["scheduler_state"])

            scaler_state = resume_ckpt.get("scaler_state")
            if scaler_state is not None:
                scaler.load_state_dict(scaler_state)

            best_loss = float(resume_ckpt.get("best_loss", best_loss))
            best_epoch = int(resume_ckpt.get("best_epoch", best_epoch))
            start_epoch = int(resume_ckpt.get("epoch", 0)) + 1
            print(f"Resumed pretrain from: {resume_path} (next epoch: {start_epoch})")
        else:
            print("Resume requested but checkpoint was not found. Starting fresh training.")

    for epoch in range(start_epoch, int(cfg.optim.epochs) + 1):
        current_lr = optimizer.param_groups[0]["lr"]

        train_stats = train_one_epoch_pretrain(
            encoder=encoder,
            projector=projector,
            criterion=criterion,
            train_loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            amp_enabled=amp_enabled,
            accum_steps=int(cfg.optim.accum_steps),
            grad_clip=float(cfg.optim.grad_clip),
            print_freq=int(cfg.print_freq),
        )

        if epoch < int(cfg.optim.epochs):
            scheduler.step()

        print(
            f"[pretrain] epoch {epoch}/{cfg.optim.epochs} "
            f"loss={train_stats['loss']:.4f} lr={current_lr:.6f} "
            f"time={train_stats['time']:.1f}s"
        )

        log_wandb(
            {
                "epoch": epoch,
                "train/loss": train_stats["loss"],
                "train/lr": current_lr,
                "train/epoch_time": train_stats["time"],
            }
        )

        is_best = train_stats["loss"] < best_loss
        if is_best:
            best_loss = float(train_stats["loss"])
            best_epoch = epoch

        checkpoint = {
            "epoch": epoch,
            "encoder_state": encoder.state_dict(),
            "projector_state": projector.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "best_loss": best_loss,
            "best_epoch": best_epoch,
            "cfg": OmegaConf.to_container(cfg, resolve=True),
            "method_temp": float(cfg.method.temperature),
            "backbone": str(cfg.model.name),
        }

        save_checkpoint(checkpoint, ckpt_dir / "last.ckpt")

        if cfg.save_best and is_best:
            save_checkpoint(checkpoint, ckpt_dir / "best.ckpt")

    if run is not None:
        wandb.summary["best_loss"] = best_loss
        wandb.summary["best_epoch"] = best_epoch
        wandb.summary["run_dir"] = str(run_dir)
        wandb.finish()

    print(f"Saved checkpoints to: {ckpt_dir}")


if __name__ == "__main__":
    main()

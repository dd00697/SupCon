from __future__ import annotations

from pathlib import Path

import hydra
import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from supcon.data import build_cifar10_loaders
from supcon.losses import SupConLoss
from supcon.models import ProjectionHead, build_encoder
from supcon.train.pretrain import build_cosine_scheduler, train_one_epoch_pretrain
from supcon.utils import (
    init_wandb,
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
    ckpt_dir = run_dir / "checkpoints"

    for epoch in range(1, int(cfg.optim.epochs) + 1):
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

        checkpoint = {
            "epoch": epoch,
            "encoder_state": encoder.state_dict(),
            "projector_state": projector.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_loss": best_loss,
            "cfg": OmegaConf.to_container(cfg, resolve=True),
            "method_temp": float(cfg.method.temperature),
            "backbone": str(cfg.model.name),
        }

        save_checkpoint(checkpoint, ckpt_dir / "last.ckpt")

        if cfg.save_best and train_stats["loss"] < best_loss:
            best_loss = train_stats["loss"]
            checkpoint["best_loss"] = best_loss
            save_checkpoint(checkpoint, ckpt_dir / "best.ckpt")

    if run is not None:
        wandb.summary["best_loss"] = best_loss
        wandb.summary["run_dir"] = str(run_dir)
        wandb.finish()

    print(f"Saved checkpoints to: {ckpt_dir}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path

import hydra
import torch
import torch.nn as nn
import wandb
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from supcon.data import build_cifar10_supervised_loaders
from supcon.models import LinearHead, build_encoder
from supcon.train.ce import evaluate_ce, train_one_epoch_ce
from supcon.train.pretrain import build_cosine_scheduler
from supcon.utils import (
    init_wandb,
    load_checkpoint,
    log_wandb,
    patch_hydra_argparse,
    print_results_table,
    save_checkpoint,
    save_resolved_config,
    save_results_row,
    seed_everything,
)

patch_hydra_argparse()


@hydra.main(config_path="configs", config_name="ce", version_base=None)
def main(cfg: DictConfig) -> None:
    seed_everything(int(cfg.seed), deterministic=bool(cfg.deterministic))

    run_dir = Path(HydraConfig.get().runtime.output_dir)
    save_resolved_config(cfg, run_dir / "config_resolved.yaml")

    run_display_name = f"{cfg.run_name}_{cfg.model.name}_seed{cfg.seed}_bs{cfg.data.batch_size}"
    run = init_wandb(cfg, run_name=run_display_name)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(cfg.amp and device.type == "cuda")

    print(f"Device: {device}")

    train_loader, val_loader, test_loader = build_cifar10_supervised_loaders(cfg.data, seed=int(cfg.seed))
    selection_split = "val" if val_loader is not None else "test"
    if val_loader is None:
        print("Validation split disabled or too small. Falling back to selecting best by test accuracy.")

    encoder, feat_dim = build_encoder(cfg.model)
    linear_head = LinearHead(in_dim=feat_dim, num_classes=int(cfg.num_classes))

    encoder = encoder.to(device)
    linear_head = linear_head.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        list(encoder.parameters()) + list(linear_head.parameters()),
        lr=float(cfg.optim.lr),
        momentum=float(cfg.optim.momentum),
        weight_decay=float(cfg.optim.weight_decay),
    )
    scheduler = build_cosine_scheduler(optimizer, cfg.optim)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_acc = float("-inf")
    best_test_acc = 0.0
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
            encoder.load_state_dict(resume_ckpt["model_state"]["encoder"], strict=True)
            linear_head.load_state_dict(resume_ckpt["model_state"]["linear_head"], strict=True)
            optimizer.load_state_dict(resume_ckpt["optimizer_state"])
            scheduler.load_state_dict(resume_ckpt["scheduler_state"])

            scaler_state = resume_ckpt.get("scaler_state")
            if scaler_state is not None:
                scaler.load_state_dict(scaler_state)

            best_acc = float(resume_ckpt.get("best_acc", best_acc))
            best_test_acc = float(resume_ckpt.get("best_test_acc", best_test_acc))
            best_epoch = int(resume_ckpt.get("best_epoch", best_epoch))
            start_epoch = int(resume_ckpt.get("epoch", 0)) + 1
            selection_split = str(resume_ckpt.get("selection_split", selection_split))
            print(f"Resumed CE training from: {resume_path} (next epoch: {start_epoch})")
        else:
            print("Resume requested but checkpoint was not found. Starting fresh training.")

    for epoch in range(start_epoch, int(cfg.optim.epochs) + 1):
        current_lr = optimizer.param_groups[0]["lr"]

        train_stats = train_one_epoch_ce(
            encoder=encoder,
            linear_head=linear_head,
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

        val_stats = None
        if val_loader is not None:
            val_stats = evaluate_ce(
                encoder=encoder,
                linear_head=linear_head,
                criterion=criterion,
                data_loader=val_loader,
                device=device,
                amp_enabled=amp_enabled,
            )

        test_stats = evaluate_ce(
            encoder=encoder,
            linear_head=linear_head,
            criterion=criterion,
            data_loader=test_loader,
            device=device,
            amp_enabled=amp_enabled,
        )

        current_acc = float(val_stats["acc1"] if val_stats is not None else test_stats["acc1"])
        is_best = current_acc > best_acc
        if is_best:
            best_acc = current_acc
            best_test_acc = float(test_stats["acc1"])
            best_epoch = epoch

        if epoch < int(cfg.optim.epochs):
            scheduler.step()

        if val_stats is None:
            print(
                f"[ce] epoch {epoch}/{cfg.optim.epochs} "
                f"train_loss={train_stats['loss']:.4f} train_acc1={train_stats['acc1']:.2f} "
                f"test_loss={test_stats['loss']:.4f} test_acc1={test_stats['acc1']:.2f} "
                f"best_test_acc1={best_test_acc:.2f}"
            )
        else:
            print(
                f"[ce] epoch {epoch}/{cfg.optim.epochs} "
                f"train_loss={train_stats['loss']:.4f} train_acc1={train_stats['acc1']:.2f} "
                f"val_loss={val_stats['loss']:.4f} val_acc1={val_stats['acc1']:.2f} "
                f"test_loss={test_stats['loss']:.4f} test_acc1={test_stats['acc1']:.2f} "
                f"best_val_acc1={best_acc:.2f} best_test_at_best_val={best_test_acc:.2f}"
            )

        wandb_metrics = {
            "epoch": epoch,
            "train/loss": train_stats["loss"],
            "train/acc1": train_stats["acc1"],
            "test/loss": test_stats["loss"],
            "test/acc1": test_stats["acc1"],
            "train/lr": current_lr,
            "best/select_acc1": best_acc,
            "best/test_acc1_at_select": best_test_acc,
            "best/epoch": best_epoch,
        }
        if val_stats is not None:
            wandb_metrics["val/loss"] = val_stats["loss"]
            wandb_metrics["val/acc1"] = val_stats["acc1"]
        log_wandb(wandb_metrics)

        checkpoint = {
            "epoch": epoch,
            "model_state": {
                "encoder": encoder.state_dict(),
                "linear_head": linear_head.state_dict(),
            },
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "best_acc": best_acc,
            "best_test_acc": best_test_acc,
            "best_epoch": best_epoch,
            "selection_split": selection_split,
            "cfg": OmegaConf.to_container(cfg, resolve=True),
        }
        save_checkpoint(checkpoint, ckpt_dir / "last.ckpt")

        if cfg.save_best and is_best:
            save_checkpoint(checkpoint, ckpt_dir / "best.ckpt")

    row = {
        "method": "ce_baseline",
        "backbone": str(cfg.model.name),
        "epochs": int(cfg.optim.epochs),
        "batch_size": int(cfg.data.batch_size),
        "temp": "-",
        "top1_accuracy": round(best_test_acc, 4),
    }

    print_results_table(row)
    save_results_row(run_dir / "results.csv", run_dir / "results.json", row)

    if run is not None:
        wandb.summary.update(row)
        wandb.summary["best_epoch"] = best_epoch
        wandb.summary["best_selection_acc"] = best_acc
        wandb.summary["selection_split"] = selection_split
        wandb.summary["run_dir"] = str(run_dir)
        wandb.finish()

    print(f"Saved checkpoints to: {ckpt_dir}")
    print(f"Saved results to: {run_dir / 'results.json'} and {run_dir / 'results.csv'}")


if __name__ == "__main__":
    main()

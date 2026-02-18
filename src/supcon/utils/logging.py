from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import wandb
from omegaconf import DictConfig, OmegaConf


def init_wandb(cfg: DictConfig, run_name: str) -> Optional[wandb.sdk.wandb_run.Run]:
    if not cfg.logging.enabled or cfg.logging.mode == "disabled":
        return None

    group = None
    try:
        group = cfg.logging.group
    except Exception:
        group = None
    if group == "":
        group = None

    return wandb.init(
        project=cfg.logging.project,
        entity=cfg.logging.entity,
        mode=cfg.logging.mode,
        group=group,
        tags=list(cfg.logging.tags),
        notes=cfg.logging.notes,
        config=OmegaConf.to_container(cfg, resolve=True),
        dir=os.getcwd(),
        name=run_name,
    )


def log_wandb(metrics: Dict[str, Any], step: Optional[int] = None) -> None:
    if wandb.run is None:
        return
    wandb.log(metrics, step=step)


def save_resolved_config(cfg: DictConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, path, resolve=True)


def print_results_table(row: Dict[str, Any]) -> None:
    headers = ["method", "backbone", "epochs", "batch_size", "temp", "top1_accuracy"]
    values = [str(row.get(k, "")) for k in headers]

    widths = [max(len(h), len(v)) for h, v in zip(headers, values)]
    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
    value_line = " | ".join(v.ljust(w) for v, w in zip(values, widths))

    print(header_line)
    print("-+-".join("-" * w for w in widths))
    print(value_line)

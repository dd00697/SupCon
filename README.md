# Supervised Contrastive Learning on CIFAR-10 

Minimal, reproducible implementation of **SupCon** (Khosla et al.) on **CIFAR-10 only** with:
- SupCon pretraining (2 views)
- Linear evaluation (frozen encoder)
- Cross-Entropy supervised baseline
- Hydra config management
- Weights & Biases logging

## What is implemented

1. **CE baseline** (`train_ce.py`)
   - Standard supervised training on CIFAR-10.
2. **SupCon pretrain** (`train_pretrain.py`)
   - Multi-positive supervised contrastive loss.
   - Positives outside the log formulation.
3. **Linear eval** (`train_linear.py`)
   - Load pretrained encoder, freeze encoder, train linear head.

Augmentations:
- SupCon pretrain uses SimCLR-style CIFAR augmentations (2 random views).
- CE + linear eval use standard supervised CIFAR augmentation (RandomCrop + Flip).

## Install

```bash
pip install -e .
```

## Quickstart

### 1) Cross-Entropy baseline

```bash
python train_ce.py
```

### 2) SupCon pretrain

```bash
python train_pretrain.py
```

### 3) Linear evaluation

Use the checkpoint from pretraining run output:

```bash
python train_linear.py pretrain_ckpt=outputs/<date>/<time>_train_pretrain/checkpoints/last.ckpt
```

## Repro commands (paper-style defaults)

```bash
python train_ce.py
python train_pretrain.py
python train_linear.py pretrain_ckpt=outputs/<date>/<time>_train_pretrain/checkpoints/best.ckpt
```

## Ablation sweeps (Hydra multirun)

### Temperature sweep

```bash
python train_pretrain.py -m method.temperature=0.05,0.1,0.2
```

### Batch size sweep

```bash
python train_pretrain.py -m data.batch_size=128,256,512
```

### Projection head depth sweep

```bash
python train_pretrain.py -m model.projector.depth=1,2
```

### Pretrain epochs sweep

```bash
python train_pretrain.py -m optim.epochs=200,500
```

## Useful overrides

Disable W&B:

```bash
python train_pretrain.py logging.mode=disabled
```

Resume from `last.ckpt` in the same run dir:

```bash
python train_ce.py resume=true hydra.run.dir=outputs/<existing_run_dir>
python train_pretrain.py resume=true hydra.run.dir=outputs/<existing_run_dir>
python train_linear.py resume=true hydra.run.dir=outputs/<existing_run_dir>
```

Resume from an explicit checkpoint path:

```bash
python train_ce.py resume=true resume_ckpt=<path_to_last.ckpt>
python train_pretrain.py resume=true resume_ckpt=<path_to_last.ckpt>
python train_linear.py resume=true resume_ckpt=<path_to_last.ckpt>
```

Organize runs in W&B:

```bash
python train_ce.py logging.group=ce_baseline
python train_pretrain.py logging.group=supcon_pretrain
python train_linear.py logging.group=supcon_linear
```

Quick sanity run:

```bash
python train_pretrain.py optim.epochs=1 data.train_subset=2048 data.test_subset=1000 logging.mode=disabled
python train_ce.py optim.epochs=1 data.train_subset=2048 data.test_subset=1000 logging.mode=disabled
python train_linear.py pretrain_ckpt=<path_to_ckpt> optim.epochs=1 data.train_subset=2048 data.test_subset=1000 logging.mode=disabled
```

## Config layout

- `configs/config.yaml`: shared base
- `configs/pretrain.yaml`: SupCon pretraining pipeline
- `configs/linear.yaml`: linear evaluation pipeline
- `configs/ce.yaml`: CE baseline pipeline
- `configs/data/cifar10.yaml`: CIFAR-10 and augmentations
- `configs/model/resnet.yaml`: ResNet backbone + projector
- `configs/method/supcon.yaml`: SupCon temperature
- `configs/optim/*.yaml`: optimizer/scheduler per pipeline
- `configs/logging/wandb.yaml`: W&B controls

## Expected artifacts per run

Each run directory under `outputs/...` contains:
- `config_resolved.yaml`
- `checkpoints/last.ckpt`
- `checkpoints/best.ckpt` (if `save_best=true`)
- `results.json` and `results.csv` for `train_ce.py` and `train_linear.py`

For CE and linear evaluation, `best.ckpt` is selected by validation accuracy (`data.val_size`, default `5000`), and `results.json/csv` records test accuracy at that best-validation epoch.

## Reporting (how to write up results)

Yes: treat this repo like a small reproducibility study. A clean report usually has:
- 1 page overview (goal, setup, methods, headline results)
- 1 main results table (CE vs SupCon+Linear, mean +- std over seeds)
- 2-3 ablation tables/plots (one variable at a time)
- a short "Differences vs paper" section

Use `REPORT_TEMPLATE.md` as a starting point.

### Where your numbers come from in this repo

After each `train_ce.py` and `train_linear.py` run, Hydra writes:
- `outputs/.../results.json` (single-row dict with `method/backbone/epochs/batch_size/temp/top1_accuracy`)
- `outputs/.../results.csv` (same content)

You typically report mean +- std across seeds (e.g. `seed=42,43,44`) for:
- CE baseline: `method=ce_baseline`
- SupCon pipeline: pretrain (no accuracy) + linear eval accuracy from `train_linear.py` (`method=supcon_linear_eval`)

### Aggregating runs across seeds (copy/paste)

From the repo root, this will scan all `outputs/**/results.json` and print a grouped summary:

```bash
python - << 'PY'
import json, math
from pathlib import Path

rows = []
for p in Path("outputs").rglob("results.json"):
    try:
        rows.append(json.loads(p.read_text(encoding="utf-8")))
    except Exception:
        pass

def mean_std(xs):
    m = sum(xs) / len(xs)
    v = sum((x - m) ** 2 for x in xs) / (len(xs) - 1) if len(xs) > 1 else 0.0
    return m, math.sqrt(v)

groups = {}
for r in rows:
    k = (r.get("method"), r.get("backbone"), r.get("epochs"), r.get("batch_size"), r.get("temp"))
    groups.setdefault(k, []).append(float(r["top1_accuracy"]))

print(f"Found {len(rows)} results.json files.")
for k, accs in sorted(groups.items()):
    m, s = mean_std(accs)
    method, backbone, epochs, bs, temp = k
    print(f"{method:16s} {backbone:8s} epochs={epochs:>4} bs={bs:>4} temp={temp}  "
          f"top1={m:.2f} +- {s:.2f} (n={len(accs)})")
PY
```

### Updating this README with results

Add a section like this once you have final numbers:

```text
## Results (CIFAR-10)

All results are Top-1 test accuracy, mean +- std over seeds {42,43,44}.

| Method          | Backbone | Protocol            | Epochs | Batch | Temp | Top-1 (mean +- std) |
|----------------|----------|---------------------|--------|-------|------|---------------------|
| CE             | ResNet50 | end-to-end          | 200    | 512   |  -   | xx.xx +- x.xx       |
| SupCon + Linear| ResNet50 | pretrain + linear   | 500/100| 512   | 0.1  | xx.xx +- x.xx       |

Ablations:
- Temperature: {0.05, 0.1, 0.2} (report best)
- Batch size: {128, 256, 512} (report best)
- Projector depth: {1, 2}
```

## Differences vs paper

- Uses torchvision ResNet with CIFAR stem adjustments, not the exact original training codebase.
- Default schedules/hyperparameters are practical paper-style defaults, not guaranteed exact paper replication.
- No distributed multi-node training logic included to keep code minimal and readable.

## Notes

- Dataset is strictly `torchvision.datasets.CIFAR10`.
- `train_supcon.py` is kept as a compatibility alias for `train_pretrain.py`.

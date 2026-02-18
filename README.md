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

## Differences vs paper

- Uses torchvision ResNet with CIFAR stem adjustments, not the exact original training codebase.
- Default schedules/hyperparameters are practical paper-style defaults, not guaranteed exact paper replication.
- No distributed multi-node training logic included to keep code minimal and readable.

## Notes

- Dataset is strictly `torchvision.datasets.CIFAR10`.
- `train_supcon.py` is kept as a compatibility alias for `train_pretrain.py`.

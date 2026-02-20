# Reproducing Supervised Contrastive Learning (SupCon) on CIFAR-10

Minimal, reproducible implementation of **SupCon** (Khosla et al.) on **CIFAR-10 only** with:
- SupCon pretraining (2 views)
- Linear evaluation (frozen encoder)
- Cross-Entropy supervised baseline
- Hydra config management
- Weights & Biases logging

## Short Report (February 20, 2026)

### Abstract
This reproduction compares Supervised Contrastive Learning (SupCon) against a standard cross-entropy (CE) supervised baseline on CIFAR-10. Using a ResNet-50 backbone, SupCon pretraining followed by linear probing reaches **94.55%** top-1 accuracy, while end-to-end CE training reaches **94.36%** top-1 accuracy.

### Experimental setup

| Item | Setting |
|---|---|
| Dataset | CIFAR-10 |
| Backbone (reported runs) | ResNet-50 |
| SupCon pretrain | 500 epochs, temperature = 0.1, batch size = 2048 |
| Linear eval | 100 epochs, frozen encoder, batch size = 2048 |
| CE baseline | 200 epochs, end-to-end supervised, batch size = 512 |

Note: these report numbers use explicit Hydra overrides and are different from the repo defaults (`resnet18`, `batch_size=512`, `pretrain epochs=200`).

### Results

| Method | Backbone | Protocol | Epochs | Batch size | Top-1 (%) |
|---|---|---|---:|---:|---:|
| SupCon + Linear Eval | ResNet-50 | SupCon pretrain + frozen linear probe | 500 + 100 | 2048 | **94.55** |
| CE Baseline | ResNet-50 | End-to-end supervised | 200 | 512 | **94.36** |

| SupCon pretraining dynamics | Value |
|---|---:|
| Contrastive loss at epoch 1 (Feb 19, 2026) | 8.3206 |
| Contrastive loss at epoch 500 (Feb 20, 2026) | 6.1129 |

### Observations
- SupCon linear evaluation and CE baseline are very close (difference: **+0.19** points for SupCon + linear probe).
- SupCon pretraining shows stable optimization with a clear loss decrease across 500 epochs.

### Conclusion
On CIFAR-10 with ResNet-50, SupCon pretraining followed by linear probing matches and slightly exceeds a strong CE baseline in this reproduction.

## W& BGraphs

- Dataset is strictly `torchvision.datasets.CIFAR10`.
- `train_supcon.py` is kept as a compatibility alias for `train_pretrain.py`.

### Cross Entropy Run, Resnet 50, Batch Size 512
<img width="1627" height="727" alt="image" src="https://github.com/user-attachments/assets/6f432b6b-22f9-4730-9b57-2d87e925cc47" />

### Pretrain Run, Epochs 500, Batch Size 2048
<img width="1659" height="366" alt="image" src="https://github.com/user-attachments/assets/11f67454-0c77-4156-8012-c1120f576029" />

### Linear Probe, Epochs 100, Batch Size 2048
<img width="1647" height="745" alt="image" src="https://github.com/user-attachments/assets/acb7e773-2644-4ae2-b601-35112e83e7a3" />


## What Is Implemented

1. **CE baseline** (`train_ce.py`)
   - Standard supervised training on CIFAR-10.
2. **SupCon pretrain** (`train_pretrain.py`)
   - Multi-positive supervised contrastive loss.
   - Positives outside the log formulation.
3. **Linear eval** (`train_linear.py`)
   - Loads pretrained encoder, freezes encoder, trains a linear head.

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

Use the checkpoint from the pretraining run output:

```bash
python train_linear.py pretrain_ckpt=outputs/<date>/<time>_train_pretrain_seed<seed>/checkpoints/last.ckpt
```

## Reproduce the reported table runs (ResNet-50 setup)

```bash
python train_pretrain.py model.name=resnet50 optim.epochs=500 data.batch_size=2048 method.temperature=0.1
python train_linear.py pretrain_ckpt=outputs/<date>/<time>_train_pretrain_seed<seed>/checkpoints/best.ckpt model.name=resnet50 optim.epochs=100 data.batch_size=2048
python train_ce.py model.name=resnet50 optim.epochs=200 data.batch_size=512
```

If memory is limited, keep the same effective batch size with gradient accumulation, for example:

```bash
python train_pretrain.py model.name=resnet50 optim.epochs=500 data.batch_size=512 optim.accum_steps=4 method.temperature=0.1
```

## Repro commands (repo defaults)

```bash
python train_ce.py
python train_pretrain.py
python train_linear.py pretrain_ckpt=outputs/<date>/<time>_train_pretrain_seed<seed>/checkpoints/best.ckpt
```
```

## Useful Overrides

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

## Config Layout

- `configs/config.yaml`: shared base
- `configs/pretrain.yaml`: SupCon pretraining pipeline
- `configs/linear.yaml`: linear evaluation pipeline
- `configs/ce.yaml`: CE baseline pipeline
- `configs/data/cifar10.yaml`: CIFAR-10 and augmentations
- `configs/model/resnet.yaml`: ResNet backbone + projector
- `configs/method/supcon.yaml`: SupCon temperature
- `configs/optim/*.yaml`: optimizer/scheduler per pipeline
- `configs/logging/wandb.yaml`: W&B controls

## Expected Artifacts Per Run

Each run directory under `outputs/...` contains:
- `config_resolved.yaml`
- `checkpoints/last.ckpt`
- `checkpoints/best.ckpt` (if `save_best=true`)
- `results.json` and `results.csv` for `train_ce.py` and `train_linear.py`

For CE and linear evaluation, `best.ckpt` is selected by validation accuracy (`data.val_size`, default `5000`), and `results.json/csv` record test accuracy at that best-validation epoch.

## Planned Ablations and Further Experiments

The next step is to run paper-aligned ablations around the main SupCon sensitivity knobs.

| Ablation | Values to run | Why it matters | Command template |
|---|---|---|---|
| Temperature (`tau`) | `0.05, 0.07, 0.1, 0.2` | SupCon is sensitive to similarity scaling; often a primary tuning lever | `python train_pretrain.py -m method.temperature=0.05,0.07,0.1,0.2` |
| Batch size | `256, 512, 1024, 2048` | Contrastive methods benefit from more in-batch positives/negatives | `python train_pretrain.py -m data.batch_size=256,512,1024,2048` |
| Projection head depth | `1, 2` | Tests whether a deeper projector improves linear separability | `python train_pretrain.py -m model.projector.depth=1,2` |
| Projection output dim | `64, 128, 256` | Checks embedding capacity vs generalization tradeoff | `python train_pretrain.py -m model.projector.out_dim=64,128,256` |
| Pretrain schedule length | `200, 500, 800` epochs | Measures whether longer contrastive pretraining still improves probe quality | `python train_pretrain.py -m optim.epochs=200,500,800` |
| Backbone scale | `resnet18, resnet50` | Quantifies whether SupCon gains change with model capacity | `python train_pretrain.py -m model.name=resnet18,resnet50` |
| Multi-seed stability | `seed=42,43,44` | Reports mean/std and reduces single-run noise | `python train_ce.py -m seed=42,43,44` and same for pretrain/linear |


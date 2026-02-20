# Report Template: SupCon on CIFAR-10 (CE vs SupCon + Linear Eval)

This repo saves single-run results to `outputs/.../results.json` and `outputs/.../results.csv`.
For a paper-style report, aggregate across seeds (mean +- std).

## 1) Goal

- What are you trying to reproduce? (CE supervised baseline vs SupCon supervised contrastive pretrain + linear eval)
- What is the dataset? (CIFAR-10 only)
- What is the metric? (Top-1 test accuracy)

## 2) Experimental Contract

Constants (kept identical across comparisons):
- Dataset: CIFAR-10
- Backbone: (e.g. ResNet50)
- Image size: 32x32
- Optimizer family: SGD (+ momentum)
- Train/eval metric: Top-1 test accuracy

Randomness policy:
- Seeds: {42, 43, 44}
- Report mean +- std across seeds.

## 3) Setup (repro details)

Hardware:
- GPU: (e.g. A100 80GB)

Software:
- OS:
- Python:
- PyTorch / torchvision:
- CUDA:

Code version:
- Git commit: (run `git rev-parse HEAD`)

## 4) Methods (what you implemented)

### 4.1 Data + augmentations

- CIFAR-10 mean/std:
- CE baseline augmentation: RandomCrop(32, padding=4) + RandomHorizontalFlip
- SupCon augmentation: RandomResizedCrop(32, scale=(0.2,1.0)), flip, jitter (p=0.8), grayscale (p=0.2), blur (configurable)
- SupCon uses 2 views per sample (TwoCropTransform).

### 4.2 Model

- Backbone: torchvision ResNet (CIFAR stem: 3x3 stride1, no maxpool)
- Representation: pooled feature vector r (fc removed / Identity)
- Projection head: MLP depth {1,2}, hidden_dim=2048, out_dim=128
- z is L2-normalized before contrastive loss.

### 4.3 Losses

- CE: cross entropy on CIFAR-10 labels.
- SupCon: multi-positive supervised contrastive loss (positives = same label), full 2B x 2B similarity, self-masked, temperature tau.

### 4.4 Training protocols

CE baseline:
- Train encoder + classifier end-to-end
- Epochs:
- Batch size:
- LR / schedule:

SupCon pretrain:
- Train encoder + projector with SupCon loss
- Epochs:
- Batch size:
- Temperature:
- LR / schedule:

Linear evaluation:
- Load SupCon encoder checkpoint
- Freeze encoder, train only linear head on r
- Epochs:
- Batch size:
- LR / schedule:

## 5) Main Results (mean +- std across seeds)

Fill with your aggregated numbers.

| Method           | Backbone | Protocol          | Epochs (pre/lin or ce) | Batch | Temp | Top-1 (mean +- std) |
|-----------------|----------|-------------------|-------------------------|-------|------|---------------------|
| CE              | ResNet50 | end-to-end        |                         |       |  -   |                     |
| SupCon + Linear | ResNet50 | pretrain + linear |                         |       |      |                     |

Notes:
- CE number comes from the best checkpoint during training (as tracked by this repo).
- SupCon + Linear number comes from `train_linear.py` result (best linear eval test acc).

## 6) Ablations

Keep everything fixed and change one variable at a time.

### 6.1 Temperature sweep

- tau in {0.05, 0.1, 0.2}

| tau  | Top-1 (mean +- std) |
|------|----------------------|
| 0.05 |                      |
| 0.10 |                      |
| 0.20 |                      |

### 6.2 Batch size sweep

- batch size in {128, 256, 512} (or larger if used)

| batch size | Top-1 (mean +- std) |
|------------|----------------------|
| 128        |                      |
| 256        |                      |
| 512        |                      |

### 6.3 Projection head depth

- depth in {1, 2}

| depth | Top-1 (mean +- std) |
|-------|----------------------|
| 1     |                      |
| 2     |                      |

## 7) Discussion

- Did SupCon + Linear beat CE for your chosen backbone/budget?
- How sensitive were results to temperature and batch size?
- Any stability issues (loss spikes, divergence, etc.)?

## 8) Differences vs the paper (and why)

Examples you can mention (only if true for your runs):
- Different backbone (ResNet18 vs ResNet50)
- Different training budget (fewer epochs)
- Single-machine (no multi-node)
- Implementation choices (torchvision ResNet + CIFAR stem)

## 9) Repro commands (exact)

Paste the exact commands you ran (seeded), plus any overrides.

Example:

```bash
# CE baseline (3 seeds)
python train_ce.py device=cuda logging.mode=online logging.group=ce_baseline model.name=resnet50 data.batch_size=512 seed=42
python train_ce.py device=cuda logging.mode=online logging.group=ce_baseline model.name=resnet50 data.batch_size=512 seed=43
python train_ce.py device=cuda logging.mode=online logging.group=ce_baseline model.name=resnet50 data.batch_size=512 seed=44

# SupCon pretrain (3 seeds)
python train_pretrain.py device=cuda logging.mode=online logging.group=supcon_pretrain model.name=resnet50 data.batch_size=512 method.temperature=0.1 seed=42
python train_pretrain.py device=cuda logging.mode=online logging.group=supcon_pretrain model.name=resnet50 data.batch_size=512 method.temperature=0.1 seed=43
python train_pretrain.py device=cuda logging.mode=online logging.group=supcon_pretrain model.name=resnet50 data.batch_size=512 method.temperature=0.1 seed=44

# Linear eval (use matching checkpoints)
python train_linear.py device=cuda logging.mode=online logging.group=supcon_linear model.name=resnet50 data.batch_size=512 method.temperature=0.1 seed=42 pretrain_ckpt=<path>
python train_linear.py device=cuda logging.mode=online logging.group=supcon_linear model.name=resnet50 data.batch_size=512 method.temperature=0.1 seed=43 pretrain_ckpt=<path>
python train_linear.py device=cuda logging.mode=online logging.group=supcon_linear model.name=resnet50 data.batch_size=512 method.temperature=0.1 seed=44 pretrain_ckpt=<path>
```


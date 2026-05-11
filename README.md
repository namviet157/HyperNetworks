# HyperNetworks

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](#part-1--static-hypernetworks-tensorflow-vision)
[![PyTorch](https://img.shields.io/badge/PyTorch-dynamic-EE4C2C?logo=pytorch&logoColor=white)](#part-2--dynamic-hypernetworks-pytorch-sequence-modeling)

This repository studies **hypernetworks** along two complementary lines:

1. **Static hypernetworks (TensorFlow 2)** — convolution layers whose kernels are generated from learned embeddings for small-scale image classification.
2. **Dynamic hypernetworks (PyTorch)** — **HyperLSTM-style** sequence modeling experiments (PTB, enwik8, Shakespeare) with training/eval/ablation scripts.

The two tracks use **different stacks** (TensorFlow vs PyTorch). Install and run the part you need using the sections below.

<p align="center">
  <img src="assets/hypernetwork.gif" width="49%" alt="Static hypernetwork illustration (1)" />
  <img src="assets/hypernetwork2.gif" width="49%" alt="Static hypernetwork illustration (2)" />
</p>
<p align="center">
  <i>Static hypernetwork idea: generate convolution kernels from learned embeddings.</i>
</p>

## Contents

- [Part 1 — Static hypernetworks (TensorFlow, vision)](#part-1--static-hypernetworks-tensorflow-vision)
  - [Static hypernetwork visuals](#static-hypernetwork-visuals)
  - [Installation (TensorFlow)](#installation-tensorflow)
  - [Datasets (vision)](#datasets-vision)
  - [Vision models](#vision-models)
  - [Running experiments (TensorFlow)](#running-experiments-tensorflow)
  - [Checkpoints and evaluation (TensorFlow)](#checkpoints-and-evaluation-tensorflow)
- [Part 2 — Dynamic hypernetworks (PyTorch, sequence modeling)](#part-2--dynamic-hypernetworks-pytorch-sequence-modeling)
  - [Installation (PyTorch)](#installation-pytorch)
  - [Quick start (PyTorch)](#quick-start-pytorch)
- [Repository layout (overview)](#repository-layout-overview)

---

## Part 1 — Static hypernetworks (TensorFlow, vision)

TensorFlow 2 code for **static hypernetworks** in image classification: baseline CNN / ResNet-style models, hypernetwork-based convolutions, a custom training loop, and controlled experiments on compact vision datasets.

### Static hypernetwork visuals

Quick visuals from the repo’s experiments and notes:

<p align="center">
  <img src="assets/hyper_tuning_mnist.png" alt="Hyper tuning on MNIST" />
</p>
<p align="center">
  <i>Hyperparameter Tuning on MNIST: Baseline vs. Hypernetwork</i>
</p>

<p align="center">
  <img src="assets/hyper_tuning_fashion_mnist.png" alt="Hyper tuning on Fashion-MNIST" />
</p>
<p align="center">
  <i>Hyperparameter Tuning on Fashion-MNIST: Baseline vs. Hypernetwork</i>
</p>

### What this part covers

- Standard convolutions and hypernetwork-generated convolutions
- A lightweight `SimpleCNN` baseline
- A ResNet-v2 style `resnet50`
- A CIFAR-style `wrn40_2` (WideResNet-40-2) with `BasicBlock`
- Training, validation, checkpointing, TensorBoard logging, and test evaluation
- Reporting from the **best validation checkpoint**

### Installation (TensorFlow)

```powershell
python -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` pins: `tensorflow==2.15.0`, `numpy==1.26.4`, `matplotlib==3.8.4`, `tensorboard==2.15.1`, `scipy`.

### Datasets (vision)

Supported datasets:

- `mnist`, `fashion_mnist`, `cifar10` — via `tf.keras.datasets`
- `svhn` — `.mat` files under `../data/svhn` by default (`train_32x32.mat`, `test_32x32.mat`)

Loaders live in `static/my_datasets/`. Labels are one-hot; images are normalized to `[0, 1]`.

### Vision models

| Model | Summary |
|--------|---------|
| `simplecnn` | Two conv layers (the second can be `HyperConv2D`), pooling, dense head; grayscale `28×28×1` or RGB `32×32×3` |
| `resnet50` | Subclassed ResNet-v2 with `BottleneckBlock` |
| `wrn40_2` | WideResNet-40-2: 3×3 stem, 16 channels, 3 stages, 6 `BasicBlock`s per stage, widths 32→64→128 |

Hypernetwork pieces are in `static/model/utils.py`: `HyperConv2D`, `SharedHyperConvMLP`. With `hyper_mode=True`, selected layers use hyper-convolutions instead of plain `Conv2D`.

### Running experiments (TensorFlow)

**Notebook:** open `static/static_hypernetwork.ipynb`.

**CLI grid runner (recommended):**

```powershell
# Train baseline + hyper side-by-side (same grid)
python static/static_hypernetwork.py train --datasets cifar10 --models wrn40_2 --hyper-modes both

# Evaluate the best validation checkpoint on test split
python static/static_hypernetwork.py eval --datasets cifar10 --models wrn40_2 --hyper-modes both
```

The underlying training/evaluation logic lives in `static/solve/static_hypernet.py` (class `Solver`).

Useful defaults: batch size `1024`, Adam, initial LR `5e-4`, exponential decay `0.99`, global gradient clip norm `100.0`, seed `42`.

**TensorBoard:**

```powershell
tensorboard --logdir runs
```

Then open [http://localhost:6006](http://localhost:6006).

### Checkpoints and evaluation (TensorFlow)

Under `save_dir`: `latest/`, `best/`, `history/`, `training_state.json`. After training, the solver can reload the **best validation** checkpoint and report test metrics. `REFERENCE_TEST_METRICS` includes a small reference table (e.g. CIFAR-10 + WRN-40-2 baseline); exact matching depends on augmentation, splits, schedules, etc.

**Practical notes:** Keras may show `Output Shape: multiple` for subclassed models; TF 2.15 can emit internal deprecation warnings; SVHN needs local `.mat` files; `wrn40_2` is the natural CIFAR-style choice over `resnet50` for many setups.

**Suggested starting points:** `simplecnn` on `mnist` for a quick sanity check; `wrn40_2` on `cifar10` for main experiments; compare `hyper_mode=False` vs `True` under identical settings.

---

## Part 2 — Dynamic hypernetworks (PyTorch, sequence modeling)

PyTorch experiments inspired by [*HyperNetworks*](https://arxiv.org/abs/1609.09106), focused on **HyperLSTM-style** dynamic modulation for character/byte-level modeling (e.g., PTB, enwik8, Shakespeare). This track includes training scripts, evaluation, ablations, and plotting utilities under `dynamic/`.

### Layout (PyTorch)

| Path | Role |
|------|------|
| `dynamic/train_all_ptb_unified.py` | Train PTB variants (baseline/LN/HyperLSTM, etc.) |
| `dynamic/train_all_enwik8.py` | Train enwik8 variants |
| `dynamic/train_shakespeare.py` | Train Shakespeare baselines |
| `dynamic/train_shakespeare_hyper.py` | Train Shakespeare HyperLSTM variants |
| `dynamic/eval_all.py` | Evaluate all datasets and write summaries |
| `dynamic/eval_ablation.py` | Ablation study runner |
| `dynamic/plot_training_curves.py` | Plot training curves from logs |
| `dynamic/plot_evaluation.py` | Plot evaluation charts |
| `dynamic/data/` | Dataset files/splits (e.g., PTB) |
| `dynamic/model_weights/` | Saved checkpoints (organized by dataset) |
| `dynamic/logs/` | Training logs/progress logs |
| `dynamic/eval_results_summary.csv` | Aggregated evaluation summary |
| `dynamic/ablation_summary.csv` | Aggregated ablation summary |

### Installation (PyTorch)

Use a **separate** virtualenv; do not reuse the TensorFlow `requirements.txt` for this track.

```bash
python -m venv venv-pt
# Windows: .\venv-pt\Scripts\activate
pip install "torch>=1.12"
```

### Quick start (PyTorch)

From the repo root:

```bash
cd dynamic
python train_shakespeare_hyper.py
python eval_all.py
python plot_evaluation.py
```

For more dataset-specific instructions (PTB/enwik8, checkpoints), see `dynamic/README.md`.

---

## Repository layout (overview)

```text
HyperNetworks/
├── assets/                   # README visuals (GIFs/figures)
├── documentations/           # Paper PDF(s)
│   └── 1609.09106v4.pdf
├── static/                   # TensorFlow 2: static hypernetworks (vision)
│   ├── model/                # CNN / ResNet / HyperConv2D utils
│   ├── my_datasets/          # MNIST/Fashion-MNIST/CIFAR-10/SVHN loaders
│   ├── solve/                # Training/eval (Solver)
│   ├── utils/                # Visualization helpers
│   ├── static_hypernetwork.py
│   ├── static_hypernetwork.ipynb
│   └── run_static_hypernetwork.sh
├── dynamic/                  # PyTorch: dynamic hypernetworks (HyperLSTM-style)
│   ├── README.md
│   ├── data/                 # PTB splits, etc.
│   ├── logs/                 # Training logs
│   ├── model_weights/         # Checkpoints (by dataset)
│   ├── evaluation_charts/     # Generated plots
│   ├── eval_ablation_results/ # Ablation outputs
│   ├── train_*.py             # Training entrypoints (PTB/enwik8/Shakespeare)
│   ├── eval_*.py              # Evaluation/ablation
│   ├── plot_*.py              # Plotting utilities
│   ├── eval_results_summary.csv
│   └── ablation_summary.csv
├── requirements.txt          # pip deps for TensorFlow (static track)
└── README.md
```
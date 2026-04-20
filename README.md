# HyperNetworks

TensorFlow 2 project for studying **static hypernetworks** in image classification.  
The repository contains baseline CNN / ResNet-style models and hypernetwork-based convolution layers that generate weights from learned embeddings, plus a custom training loop for running controlled experiments on small vision datasets.

## What This Project Covers

- Standard convolutions and hypernetwork-generated convolutions
- A lightweight `SimpleCNN` baseline
- A ResNet-v2 style `resnet50`
- A CIFAR-style `wrn40_2` (WideResNet-40-2) with `BasicBlock`
- Training, validation, checkpointing, TensorBoard logging, and test evaluation
- Reproducibility reporting from the **best validation checkpoint**

## Installation

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` currently pins:

- `tensorflow==2.15.0`
- `numpy==1.26.4`
- `matplotlib==3.8.4`
- `tensorboard==2.15.1`
- `scipy`

## Datasets

Supported datasets:

- `mnist`
- `fashion_mnist`
- `cifar10`
- `svhn`

Dataset loaders live in `my_datasets/`.

### Notes

- `mnist`, `fashion_mnist`, and `cifar10` are loaded from `tf.keras.datasets`
- `svhn` expects `.mat` files at `../data/svhn` by default:
  - `train_32x32.mat`
  - `test_32x32.mat`
- Labels are converted to one-hot vectors in the dataset loaders
- Images are normalized to `[0, 1]`

## Models

### `simplecnn`

A compact 2-convolution classifier:

- `conv1`: standard `Conv2D`
- `conv2`: standard `Conv2D` or `HyperConv2D`
- max pooling after each convolution
- dense classifier head

This model works with both grayscale (`28x28x1`) and RGB (`32x32x3`) inputs.

### `resnet50`

A Keras subclassed ResNet-v2 style network built from `BottleneckBlock`.

### `wrn40_2`

A WideResNet-40-2 style network for CIFAR-sized inputs:

- stem `3x3` convolution with 16 channels
- 3 stages
- 6 `BasicBlock`s per stage
- widths `32 -> 64 -> 128`
- downsampling in the first block of stages 2 and 3

## Hypernetwork Components

The main hypernetwork pieces are implemented in `model/utils.py`:

- `HyperConv2D`: generates convolution kernels from learned block embeddings
- `SharedHyperConvMLP`: shared two-layer MLP used across multiple hyper-convolution layers

When `hyper_mode=True`, supported models replace selected convolution layers with hypernetwork-based convolutions instead of standard `Conv2D`.

## Project Structure

```text
HyperNetworks/
|- model/
|  |- simple_cnn.py
|  |- resnet.py
|  |- utils.py
|  `- nets/
|     |- resnet_utils.py
|     `- resnet_v2.py
|- my_datasets/
|  |- mnist.py
|  |- fashion_mnist.py
|  |- cifar10.py
|  `- svhn.py
|- solve/
|  `- static_hypernet.py
|- utils/
|  `- visualize.py
|- static_hypernetwork.ipynb
`- requirements.txt
```

## Running Experiments

### Option 1: Notebook

Open `static_hypernetwork.ipynb` and run the cells for the dataset / model combination you want.

The notebook already includes helper code such as:

- random seed setup
- output path generation under `runs/`
- example training runs for baseline and hypernetwork variants

### Option 2: Use `Solver` Directly

Example:

```python
from pathlib import Path

from solve.static_hypernet import Solver


def build_run_paths(dataset, model, hyper_mode=False):
    run_name = f"{dataset}_{model}"
    if hyper_mode:
        run_name += "_hyper"
    run_root = Path("runs") / run_name
    return {
        "logpath": str(run_root / "logs"),
        "save_dir": str(run_root / "checkpoints"),
    }


run_paths = build_run_paths("cifar10", "wrn40_2", hyper_mode=True)

solver = Solver(
    dataset="cifar10",
    model="wrn40_2",
    max_epoch=20,
    hyper_mode=True,
    logpath=run_paths["logpath"],
    save_dir=run_paths["save_dir"],
    show_sample=False,
    show_filters=False,
)
solver.train()
```

## `Solver` Configuration

Main constructor arguments in `solve/static_hypernet.py`:

- `dataset`: `mnist`, `fashion_mnist`, `cifar10`, or `svhn`
- `model`: `simplecnn`, `resnet50`, or `wrn40_2`
- `max_epoch`: total training epochs
- `hyper_mode`: enable hypernetwork convolutions where supported
- `logpath`: TensorBoard output directory
- `save_dir`: checkpoint directory
- `val_split`: validation split ratio, default `0.1`
- `resume`: resume from the latest checkpoint
- `eval_only`: skip training and evaluate saved weights
- `show_sample`: visualize a training image
- `show_filters`: visualize the first convolution filters when available
- `run_final_test_from_best`: after training, reload the best validation checkpoint and report final test metrics

Important built-in defaults:

- batch size: `1024`
- optimizer: `Adam`
- initial learning rate: `5e-4`
- exponential decay: `0.99`
- gradient clipping: global norm `100.0`
- seed: `42`

## Checkpoints and Logging

Each run writes artifacts under the `save_dir` you provide:

- `latest/`: rolling checkpoint for continuing interrupted training
- `best/`: best checkpoint based on validation accuracy
- `history/`: optional history checkpoints
- `training_state.json`: metadata such as completed epoch and best metric

TensorBoard logs are written to `logpath`.

To inspect them:

```powershell
tensorboard --logdir runs
```

Then open [http://localhost:6006](http://localhost:6006).

## Evaluation Protocol

The training loop tracks:

- train loss / accuracy
- validation loss / accuracy
- test loss / accuracy
- learning rate

After training, the solver can automatically:

1. reload the checkpoint stored by `best_manager`
2. evaluate on the **test split**
3. print final **test loss** and **test accuracy (%)**

When `eval_only=True`, the solver restores the **best checkpoint** and reports evaluation metrics without further training.

## Reproducibility Notes

The solver includes a small reference table for published results in `REFERENCE_TEST_METRICS`.

At the moment, the built-in paper comparison covers:

- `cifar10` + `wrn40_2` + `hyper_mode=False`

Reference used:

- Zagoruyko and Komodakis, *Wide Residual Networks*, arXiv:1605.07146, Table 4

Important: direct metric matching is not guaranteed unless your setup matches the paper, including:

- preprocessing / normalization
- data augmentation
- validation split policy
- learning-rate schedule
- number of epochs
- checkpoint selection rule

## Known Practical Notes

- Keras may display `Output Shape: multiple` in `model.summary()` for subclassed models; this is expected behavior
- TensorFlow / Keras 2.15 can print deprecation warnings from internal APIs; these do not necessarily come from your own model code
- `SVHN` requires local `.mat` files and is not auto-downloaded in the current implementation
- `resnet50` exists for experimentation, but `wrn40_2` is the more natural CIFAR-style architecture in this repo

## Suggested Starting Points

- Use `simplecnn` on `mnist` to verify the training loop quickly
- Use `wrn40_2` on `cifar10` for the main WideResNet-style experiments
- Compare `hyper_mode=False` vs `hyper_mode=True` under the same run settings

## License / Citation

If you use this repository in a report or thesis, cite the original papers behind the implemented ideas, especially:

- HyperNetworks
- Wide Residual Networks

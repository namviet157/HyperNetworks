# StaticHyperNetwork

StaticHyperNetwork is a TensorFlow-based image classification project for experimenting with standard convolution layers and static hypernetwork-generated convolution weights.

## Features

- Train on `mnist` or `cifar10`
- Choose between `simplecnn` and `resnet50`
- Optional `--hyper-mode` for hypernetwork-based convolutions
- Native TensorFlow 2 / Keras models and layers
- Custom training loop based on `tf.GradientTape`
- Validation split during training
- TensorBoard logging for loss, accuracy, learning rate, and best metric
- Checkpoint saving for latest, best, and optional epoch history
- Resume training and run evaluation from saved checkpoints

## Installation

Create and activate a virtual environment, then install the pinned dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Training

Minimal training run:

```powershell
python main.py
```

Example with explicit outputs:

```powershell
python main.py `
  --dataset mnist `
  --model simplecnn `
  --epochs 5 `
  --hyper-mode
```

The launcher now keeps most training settings inside `main.py` as defaults, and automatically writes outputs to:

- `runs/<dataset>_<model>/logs`
- `runs/<dataset>_<model>/checkpoints`

If `--hyper-mode` is enabled, the run folder becomes `runs/<dataset>_<model>_hyper`.

Enable the hypernetwork path:

```powershell
python main.py --dataset mnist --model simplecnn --hyper-mode
```

Visualize one training sample before running and the first convolution filters after training:

```powershell
python main.py --show-sample --show-filters
```

## Resume Training

To continue from the latest checkpoint in `--save-dir`:

```powershell
python main.py `
  --dataset mnist `
  --model simplecnn `
  --epochs 10 `
  --resume
```

The training state is tracked in `training_state.json`, so the solver can continue from the next unfinished epoch and keep the previous best metric. Make sure the `dataset`, `model`, and `hyper-mode` flags match the run you want to resume.

## Evaluation Only

Run evaluation from the latest saved checkpoint without additional training:

```powershell
python main.py `
  --dataset mnist `
  --model simplecnn `
  --eval-only
```

## Checkpoint Layout

Inside the auto-generated checkpoint directory, the training workflow creates:

- `latest/`: rolling checkpoint used by `--resume` and `--eval-only`
- `best/`: checkpoint with the best validation accuracy, or best test accuracy if validation is disabled
- `history/`: per-epoch checkpoints when `--save-best-only` is not enabled
- `training_state.json`: saved metadata for completed epochs and the current best metric

If you only want the rolling and best checkpoints, add:

```powershell
python main.py --save-best-only
```

## TensorBoard

Launch TensorBoard against the generated log directory:

```powershell
tensorboard --logdir runs
```

Then open [http://localhost:6006](http://localhost:6006).

## Command Line Arguments

- `--dataset`: `mnist` or `cifar10`
- `--model`: `simplecnn` or `resnet50`
- `--epochs`: total number of epochs to train
- `--hyper-mode`: enable hypernetwork-generated convolution weights
- `--resume`: restore the latest checkpoint and continue training
- `--eval-only`: restore the latest checkpoint and run evaluation only
- `--show-sample`: display the first training image before training/evaluation
- `--show-filters`: display the first convolution filters after training or evaluation

## Notes

- `simplecnn` now supports both grayscale and RGB inputs.
- `resnet50` now uses a Keras ResNet-v2 style implementation instead of the old `tf_slim` graph code.
- If `--val-split 0` is used, best checkpoint selection falls back to test accuracy.
- Advanced defaults such as batch size, learning rate, validation split, and seed are centralized in `main.py`.
- The repository ignores generated logs and checkpoint artifacts via `.gitignore`.

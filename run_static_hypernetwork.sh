#!/usr/bin/env bash
# Run static hypernetwork experiments with different CLI argument sets.
# Usage: from repo root, bash run_static_hypernetwork.sh [verify|quick|...]
# On Windows: Git Bash, WSL, or MSYS2.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# Prefer python3; fall back to python (Windows/Git Bash often has `python` only).
if command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  PY=python
fi

case "${1:-help}" in
  verify)
    # Sanity-check grid sizes (no training).
    "$PY" static_hypernetwork.py verify --datasets all --models all --hyper-modes both
    ;;
  quick-train)
    # One small MNIST baseline run (override epochs for a fast smoke test).
    "$PY" static_hypernetwork.py train \
      --datasets mnist \
      --models simplecnn \
      --hyper-modes baseline \
      --max-epoch 1 \
      --batch-size 512
    ;;
  quick-eval)
    # Evaluate best checkpoint for the same narrow grid (requires prior train).
    "$PY" static_hypernetwork.py eval \
      --datasets mnist \
      --models simplecnn \
      --hyper-modes baseline \
      --results-json runs/evaluation_mnist_quick.json
    ;;
  train-full)
    # Full 24-run main grid (long): 4 datasets x 3 models x 2 hyper modes.
    "$PY" static_hypernetwork.py train \
      --datasets all \
      --models all \
      --hyper-modes both \
      --setting-name main
    ;;
  eval-full)
    "$PY" static_hypernetwork.py eval \
      --datasets all \
      --models all \
      --hyper-modes both \
      --setting-name main \
      --results-json runs/evaluation_results.json
    ;;
  benchmark-full)
    # 48 runs: full grid x 2 benchmark hyperparameter presets (long).
    "$PY" static_hypernetwork.py benchmark \
      --datasets all \
      --models all \
      --hyper-modes both \
      --results-json runs/benchmark_results.json
    ;;
  train-subset-cifar-svhn)
    "$PY" static_hypernetwork.py train \
      --datasets cifar10,svhn \
      --models wrn40_2,resnet50 \
      --hyper-modes both \
      --setting-name main
    ;;
  eval-subset-cifar-svhn)
    "$PY" static_hypernetwork.py eval \
      --datasets cifar10,svhn \
      --models wrn40_2,resnet50 \
      --hyper-modes both \
      --setting-name main \
      --results-json runs/evaluation_cifar_svhn_wrns_resnet.json
    ;;
  help|*)
    cat <<'EOF'
run_static_hypernetwork.sh — wrapper around static_hypernetwork.py

  bash run_static_hypernetwork.sh verify
  bash run_static_hypernetwork.sh quick-train
  bash run_static_hypernetwork.sh quick-eval
  bash run_static_hypernetwork.sh train-full
  bash run_static_hypernetwork.sh eval-full
  bash run_static_hypernetwork.sh benchmark-full
  bash run_static_hypernetwork.sh train-subset-cifar-svhn
  bash run_static_hypernetwork.sh eval-subset-cifar-svhn

Direct Python examples:

  python static_hypernetwork.py verify
  python static_hypernetwork.py train --datasets mnist --models simplecnn --hyper-modes hyper --max-epoch 5
  python static_hypernetwork.py eval --datasets all --models all --hyper-modes both --results-json runs/out.json
  python static_hypernetwork.py benchmark --datasets fashion_mnist --models simplecnn --hyper-modes both
EOF
    ;;
esac

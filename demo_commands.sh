#!/usr/bin/env bash

set -euo pipefail

# Quick demo on the built-in corpus.
python run_char_experiment.py train \
  --model hyperlstm \
  --device cpu \
  --output-dir artifacts/hyperlstm_quick_demo \
  --steps 120 \
  --eval-every 30 \
  --sample-every 60

# Stronger demo on Tiny Shakespeare.
python compare_models.py \
  --download-tinyshakespeare \
  --device cpu \
  --output-dir artifacts/shakespeare_comparison \
  --steps 300 \
  --eval-every 50 \
  --sample-every 150 \
  --prompt "ROMEO:"

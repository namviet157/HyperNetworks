# HyperNetworks (arXiv:1609.09106)

## Setup

### 1. Download Model Weights
Download model weights from Google Drive folder and place into these folders:

```
results_ptb_final/         <- PTB checkpoints (4 files)
results_enwik8_final/      <- enwik8 checkpoints (4 files)
train_shakespeare_results/ <- Shakespeare checkpoints (4 files)
```

Required checkpoint files:

```
results_ptb_final/
├── LSTM_Baseline_1000_best.pth
├── LayerNorm_LSTM_1000_best.pth
├── HyperLSTM_1000_best.pth
└── LayerNorm_HyperLSTM_1000_best.pth

results_enwik8_final/
├── LSTM_no_dropout_best.pth
├── LayerNorm_LSTM_best.pth
├── HyperLSTM_best.pth
└── LayerNorm_HyperLSTM_best.pth

train_shakespeare_results/
├── Baseline_LSTM_best.pth
├── LN_LSTM_best.pth
├── Hyper_LSTM_best.pth
└── LN_Hyper_LSTM_best.pth
```

### 2. Download enwik8 Data (optional, ~100MB)
```bash
python3 -c "
import urllib.request, zipfile
urllib.request.urlretrieve('https://data.deepai.org/enwik8.zip', 'enwik8.zip')
with zipfile.ZipFile('enwik8.zip', 'r') as z: z.extractall()
import os; os.remove('enwik8.zip')
"
```
This downloads `enwik8` to the root directory (where the scripts expect it).

Shakespeare data is auto-downloaded on first run.

## Training

```bash
# PTB (GPU 4)
CUDA_VISIBLE_DEVICES=4 python3 train_all_ptb_unified.py

# enwik8 (GPU 5)
CUDA_VISIBLE_DEVICES=5 python3 train_all_enwik8.py

# Shakespeare (GPU 7)
CUDA_VISIBLE_DEVICES=7 python3 train_shakespeare.py
CUDA_VISIBLE_DEVICES=7 python3 train_shakespeare_hyper.py

# Ablation Study (GPU 3)
CUDA_VISIBLE_DEVICES=3 python3 eval_ablation.py
```

## Evaluation

```bash
# All datasets + ablation
CUDA_VISIBLE_DEVICES=4 python3 eval_all.py

# Plot charts
python3 plot_evaluation.py
```

## Results Summary

### PTB (Test BPC — lower is better)
| Model | Test BPC |
|-------|----------|
| LSTM Baseline | 0.8937 |
| LayerNorm LSTM | 0.8795 |
| HyperLSTM | 0.9009 |
| LayerNorm HyperLSTM | 0.8831 |

### enwik8 (Test BPC — lower is better)
| Model | Test BPC |
|-------|----------|
| LSTM (no dropout) | 1.2519 |
| LayerNorm LSTM | 1.1791 |
| HyperLSTM | 1.2193 |
| LayerNorm HyperLSTM | 1.6323 |

### Shakespeare (Test BPC — lower is better)
| Model | Test BPC |
|-------|----------|
| Baseline LSTM | 1.5769 |
| LN LSTM | 1.5693 |
| Hyper LSTM | 1.6054 |
| LN Hyper LSTM | 1.5467 |

See `eval_results_summary.csv` and `ablation_summary.csv` for full results.

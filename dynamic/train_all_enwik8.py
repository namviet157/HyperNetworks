"""
HyperLSTM Character-Level Language Model on enwik8 (Hutter Prize Wikipedia)
Paper: "HyperNetworks" (Ha et al., 2016) - arXiv:1609.09106
Table 4 reference (hidden=1800):
  - LSTM 1800 no recurrent dropout:    Test BPC ~1.470
  - LayerNorm LSTM 1800:                Test BPC ~1.402
  - HyperLSTM 1800:                     Test BPC ~1.391
  - LayerNorm HyperLSTM 1800:           Test BPC ~1.353
  - LayerNorm HyperLSTM 2048:           Test BPC ~1.340

We use H=1000, HH=128, HE=64 for speed (~2h/model on A100).
"""

import os
import time
import datetime
import math
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

OUTPUT_DIR = "results_enwik8_final"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

LOG_FILE = os.path.join(OUTPUT_DIR, "progress_log.txt")

def write_log(msg):
    print(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


# =====================================================================
# BASELINE LSTM CELL (paper Eq. 9, Appendix A.2.2)
# =====================================================================
class BaselineLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_keep=0.9, use_ln=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_ln = use_ln
        self.dropout = nn.Dropout(1 - dropout_keep) if dropout_keep < 1.0 else nn.Identity()
        self.W = nn.Linear(input_size + hidden_size, 4 * hidden_size)

        if self.use_ln:
            self.ln_i = nn.LayerNorm(hidden_size)
            self.ln_f = nn.LayerNorm(hidden_size)
            self.ln_o = nn.LayerNorm(hidden_size)
            self.ln_g = nn.LayerNorm(hidden_size)
            self.ln_c = nn.LayerNorm(hidden_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        device = x.device
        h = torch.zeros(batch_size, self.hidden_size, device=device)
        c = torch.zeros(batch_size, self.hidden_size, device=device)

        outputs = []
        for t in range(seq_len):
            gates = self.W(torch.cat([x[:, t, :], h], dim=1))
            i, f, o, g = gates.chunk(4, 1)

            if self.use_ln:
                i = self.ln_i(i)
                f = self.ln_f(f)
                o = self.ln_o(o)
                g = self.ln_g(g)

            i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
            g = torch.tanh(g)

            # Recurrent Dropout: dropout on gate g (paper Eq. 13)
            c = f * c + i * self.dropout(g)
            c_norm = self.ln_c(c) if self.use_ln else c
            h = o * torch.tanh(c_norm)
            outputs.append(h)

        return torch.stack(outputs, dim=1), (h, c)


# =====================================================================
# HYPERLSTM CELL (paper Eq. 10-13, Appendix A.2.2 & A.2.3)
# =====================================================================
class HyperLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, hyper_hidden_size=128,
                 hyper_embed_size=4, dropout_keep=0.9, use_ln=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.hyper_hidden_size = hyper_hidden_size
        self.hyper_embed_size = hyper_embed_size
        self.use_ln = use_ln
        self.dropout = nn.Dropout(1 - dropout_keep) if dropout_keep < 1.0 else nn.Identity()

        # Main LSTM: bias=False (dynamic bias from hypernetwork)
        self.W = nn.Linear(input_size + hidden_size, 4 * hidden_size, bias=False)

        # HyperLSTM Cell: separate weights + LayerNorm per gate (paper Eq. 10)
        self.Wi_h = nn.Linear(hyper_hidden_size, hyper_hidden_size, bias=False)
        self.Wi_x = nn.Linear(input_size, hyper_hidden_size, bias=False)
        self.bi   = nn.Parameter(torch.zeros(hyper_hidden_size))

        self.Wg_h = nn.Linear(hyper_hidden_size, hyper_hidden_size, bias=False)
        self.Wg_x = nn.Linear(input_size, hyper_hidden_size, bias=False)
        self.bg   = nn.Parameter(torch.zeros(hyper_hidden_size))

        self.Wf_h = nn.Linear(hyper_hidden_size, hyper_hidden_size, bias=False)
        self.Wf_x = nn.Linear(input_size, hyper_hidden_size, bias=False)
        self.bf   = nn.Parameter(torch.zeros(hyper_hidden_size))

        self.Wo_h = nn.Linear(hyper_hidden_size, hyper_hidden_size, bias=False)
        self.Wo_x = nn.Linear(input_size, hyper_hidden_size, bias=False)
        self.bo   = nn.Parameter(torch.zeros(hyper_hidden_size))

        self.ln_i = nn.LayerNorm(hyper_hidden_size)
        self.ln_g = nn.LayerNorm(hyper_hidden_size)
        self.ln_f = nn.LayerNorm(hyper_hidden_size)
        self.ln_o = nn.LayerNorm(hyper_hidden_size)
        self.ln_c = nn.LayerNorm(hyper_hidden_size)

        # Projections: hyper_hidden -> embedding (paper Eq. 11)
        self.proj_z_x = nn.Linear(hyper_hidden_size, 4 * hyper_embed_size)
        self.proj_z_h = nn.Linear(hyper_hidden_size, 4 * hyper_embed_size)
        self.proj_z_b = nn.Linear(hyper_hidden_size, 4 * hyper_embed_size)

        # Weight scaling: embedding -> hidden (paper Eq. 12)
        self.proj_d_x = nn.Linear(hyper_embed_size, hidden_size, bias=False)
        self.proj_d_h = nn.Linear(hyper_embed_size, hidden_size, bias=False)
        self.proj_bias = nn.Linear(hyper_embed_size, hidden_size)

        if self.use_ln:
            self.ln_i_main = nn.LayerNorm(hidden_size)
            self.ln_f_main = nn.LayerNorm(hidden_size)
            self.ln_o_main = nn.LayerNorm(hidden_size)
            self.ln_g_main = nn.LayerNorm(hidden_size)
            self.ln_c_main = nn.LayerNorm(hidden_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        device = x.device
        h     = torch.zeros(batch_size, self.hidden_size,     device=device)
        c     = torch.zeros(batch_size, self.hidden_size,     device=device)
        h_hat = torch.zeros(batch_size, self.hyper_hidden_size, device=device)
        c_hat = torch.zeros(batch_size, self.hyper_hidden_size, device=device)

        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]

            # HyperLSTM Cell with LayerNorm (paper Eq. 10)
            i_t = self.ln_i(self.Wi_h(h_hat) + self.Wi_x(x_t) + self.bi)
            g_t = self.ln_g(self.Wg_h(h_hat) + self.Wg_x(x_t) + self.bg)
            f_t = self.ln_f(self.Wf_h(h_hat) + self.Wf_x(x_t) + self.bf)
            o_t = self.ln_o(self.Wo_h(h_hat) + self.Wo_x(x_t) + self.bo)

            i_t = torch.sigmoid(i_t)
            g_t = torch.tanh(g_t)
            f_t = torch.sigmoid(f_t)
            o_t = torch.sigmoid(o_t)

            c_hat = f_t * c_hat + i_t * g_t
            h_hat = o_t * torch.tanh(self.ln_c(c_hat))

            # Weight scaling vectors (paper Eq. 11-12)
            zx = self.proj_z_x(h_hat)
            zh = self.proj_z_h(h_hat)
            zb = self.proj_z_b(h_hat)

            dx = torch.cat([self.proj_d_x(z) for z in zx.chunk(4, -1)], dim=-1)
            dh = torch.cat([self.proj_d_h(z) for z in zh.chunk(4, -1)], dim=-1)
            bd = torch.cat([self.proj_bias(z) for z in zb.chunk(4, -1)], dim=-1)

            # Main LSTM gates
            W_x = F.linear(x_t, self.W.weight[:, :x_t.shape[-1]])
            W_h = F.linear(h,     self.W.weight[:, x_t.shape[-1]:])
            gates = W_x * dx + W_h * dh + bd
            i, f, o, g = gates.chunk(4, -1)

            if self.use_ln:
                i = self.ln_i_main(i)
                f = self.ln_f_main(f)
                o = self.ln_o_main(o)
                g = self.ln_g_main(g)

            i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
            g = torch.tanh(g)

            # Recurrent Dropout (paper Eq. 13)
            c = f * c + i * self.dropout(g)
            c_norm = self.ln_c_main(c) if self.use_ln else c
            h = o * torch.tanh(c_norm)
            outputs.append(h)

        return torch.stack(outputs, dim=1), (h, c)


# =====================================================================
# CHARACTER-LEVEL LANGUAGE MODEL
# =====================================================================
class CharLM(nn.Module):
    def __init__(self, core, vocab_size, embed_size, hidden_size, dropout_keep=0.9):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.drop  = nn.Dropout(1 - dropout_keep)
        self.core  = core
        self.out   = nn.Linear(hidden_size, vocab_size)

    def forward(self, idx):
        x = self.drop(self.embed(idx))
        out, _ = self.core(x)
        return self.out(self.drop(out))


# =====================================================================
# WEIGHT INITIALIZATION (Appendix A.2.3 & A.3.4)
# =====================================================================
def init_orthogonal(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def init_hyperlstm(model):
    """Special initialization for HyperLSTM (paper A.2.3)."""
    for name, p in model.named_parameters():
        if p.dim() >= 2:
            nn.init.orthogonal_(p)
        else:
            nn.init.zeros_(p)

    # proj_z_x, proj_z_h: weights=0, biases=1
    nn.init.zeros_(model.proj_z_x.weight)
    nn.init.ones_(model.proj_z_x.bias)
    nn.init.zeros_(model.proj_z_h.weight)
    nn.init.ones_(model.proj_z_h.bias)

    # proj_z_b: weights ~ N(0, 0.01)
    nn.init.normal_(model.proj_z_b.weight, mean=0.0, std=0.01)
    nn.init.zeros_(model.proj_z_b.bias)

    # Weight scaling vectors: 0.1 / Nz
    nn.init.constant_(model.proj_d_x.weight, 0.1 / model.hyper_embed_size)
    nn.init.constant_(model.proj_d_h.weight, 0.1 / model.hyper_embed_size)


# =====================================================================
# ENWIK8 DATA
# Train: first 90M chars | Val: next 5M | Test: last 5M
# =====================================================================
write_log("Loading enwik8 data...")

with open("enwik8", 'r', encoding='utf-8', errors='ignore') as f:
    enwik8 = f.read()

TRAIN_END = 90_000_000
VAL_END   = 95_000_000

train_text = enwik8[:TRAIN_END]
val_text   = enwik8[TRAIN_END:VAL_END]
test_text  = enwik8[VAL_END:]

chars = sorted(set(train_text + val_text + test_text))
vocab_size = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}

train_data = torch.tensor([char_to_idx[ch] for ch in train_text], dtype=torch.long)
val_data   = torch.tensor([char_to_idx[ch] for ch in val_text],   dtype=torch.long)
test_data  = torch.tensor([char_to_idx[ch] for ch in test_text],  dtype=torch.long)

write_log(f"Vocab={vocab_size} | Train={len(train_data):,} | "
          f"Val={len(val_data):,} | Test={len(test_data):,}")


def get_batch(data, seq_len, batch_size, device):
    ix = torch.randint(len(data) - seq_len, (batch_size,))
    x = torch.stack([data[i:i+seq_len]       for i in ix])
    y = torch.stack([data[i+1:i+seq_len+1]   for i in ix])
    return x.to(device), y.to(device)


def eval_model(model, data, batch_size, seq_len, device, num_batches=50):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total = 0.0
    with torch.no_grad():
        for _ in range(num_batches):
            x, y = get_batch(data, seq_len, batch_size, device)
            loss = criterion(model(x).view(-1, vocab_size), y.view(-1))
            total += loss.item()
    model.train()
    return total / num_batches


# =====================================================================
# TRAINING
# =====================================================================
def train_model(name, model, steps, batch_size, seq_len, device):
    write_log(f"\nTraining {name}...")
    model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    ckpt_path = os.path.join(OUTPUT_DIR, f"{name}.ckpt")

    start_step = 1
    best_val   = float('inf')

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(ckpt['state'])
        optimizer.load_state_dict(ckpt['opt'])
        start_step = ckpt['step'] + 1
        best_val   = ckpt['best']
        write_log(f"  Resumed from step {start_step}")

    t0 = time.time()
    t_last = t0

    for step in range(start_step, steps + 1):
        x, y = get_batch(train_data, seq_len, batch_size, device)

        optimizer.zero_grad()
        loss = criterion(model(x).view(-1, vocab_size), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 500 == 0 or step == 1:
            val_loss = eval_model(model, val_data, batch_size, seq_len, device)
            val_bpc  = val_loss / math.log(2)

            elapsed = time.time() - t_last
            speed   = 500 / elapsed if step > 1 else 0
            eta     = str(datetime.timedelta(seconds=int((steps - step) / speed))) if speed > 0 else "..."

            write_log(f"  [{name}] {step}/{steps} | Val BPC={val_bpc:.4f} | "
                      f"Val NLL={val_loss:.4f} | {speed:.1f} it/s | ETA={eta}")
            t_last = time.time()

            torch.save({
                'step': step, 'state': model.state_dict(),
                'opt': optimizer.state_dict(), 'best': best_val
            }, ckpt_path)

            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(),
                           os.path.join(OUTPUT_DIR, f"{name}_best.pth"))
                write_log(f"    -> Best! BPC={val_bpc:.4f}")

    # Final test eval
    model.load_state_dict(
        torch.load(os.path.join(OUTPUT_DIR, f"{name}_best.pth"), weights_only=False))
    test_loss = eval_model(model, test_data, batch_size, seq_len, device, num_batches=200)
    test_bpc  = test_loss / math.log(2)
    write_log(f"  [{name}] Test BPC={test_bpc:.4f} | Test NLL={test_loss:.4f}")
    return test_bpc


# =====================================================================
# MAIN
# =====================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    write_log(f"Device: {device}")

    # Hyperparameters (scaled down from paper for 12h constraint)
    # Paper: H=1800, HH=256, HE=64, seq=250, steps=many
    # We use: H=1000, HH=128, HE=64, seq=250, steps=25000
    E  = 64       # Embedding size
    H  = 1000     # Hidden size
    HH = 128      # HyperLSTM hidden (paper: 256)
    HE = 64       # Hyper embedding size (paper: 64)
    BS = 64       # Batch size
    SL = 250      # Sequence length (paper: 250)
    ST = 25000    # Total steps per model
    KP = 0.90     # Keep probability

    write_log("=" * 60)
    write_log("enwik8 Char-LM (arXiv:1609.09106 Table 4)")
    write_log(f"  embed={E}, hidden={H}, hyper_hid={HH}, hyper_emb={HE}")
    write_log(f"  batch={BS}, seq={SL}, steps={ST}, keep_prob={KP}")
    write_log(f"  Paper ref (H=1800): HyperLSTM~1.391, LN-HyperLSTM~1.353")
    write_log("=" * 60)

    csv_path = os.path.join(OUTPUT_DIR, "results.csv")
    csv_file = open(csv_path, "a", newline="")
    cw = csv.writer(csv_file)
    if os.path.getsize(csv_path) == 0:
        cw.writerow(["Model", "Step", "Val_BPC"])

    configs = [
        # NOTE: No recurrent dropout baseline per paper A.3.4
        ("LSTM_no_dropout",
         BaselineLSTM(E, H, KP, use_ln=False)),
        ("LayerNorm_LSTM",
         BaselineLSTM(E, H, KP, use_ln=True)),
        ("HyperLSTM",
         HyperLSTM(E, H, HH, HE, KP, use_ln=False)),
        ("LayerNorm_HyperLSTM",
         HyperLSTM(E, H, HH, HE, KP, use_ln=True)),
    ]

    results = {}
    for name, core in configs:
        cw.writerow([name, "START", ""])
        lm = CharLM(core, vocab_size, E, H, KP)
        lm.apply(init_orthogonal)
        if "Hyper" in name:
            init_hyperlstm(lm.core)
        test_bpc = train_model(name, lm, ST, BS, SL, device)
        results[name] = test_bpc
        csv_file.flush()

    csv_file.close()
    write_log("\n" + "=" * 60)
    write_log("SUMMARY")
    for n, bpc in results.items():
        write_log(f"  {n}: {bpc:.4f} BPC")
    write_log(f"\nCSV: {csv_path}")
    write_log(f"Log: {LOG_FILE}")
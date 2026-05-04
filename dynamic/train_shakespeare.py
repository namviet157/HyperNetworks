"""
Training: Shakespeare (Novel Dataset) - Character Level LM
Trains 4 models: Baseline, +LN, +Hyper, +LN+Hyper
Dataset: Tiny Shakespeare from Karpathy's char-rnn

Run:  CUDA_VISIBLE_DEVICES=0 python3 train_shakespeare.py
Output: train_shakespeare_results/
"""

import os, time, datetime, math, csv, urllib.request, torch, torch.nn as nn, torch.optim as optim

OUT = "train_shakespeare_results"
os.makedirs(OUT, exist_ok=True)
LOG = os.path.join(OUT, "log.txt")

def log(msg):
    print(msg)
    with open(LOG, "a") as f: f.write(msg + "\n")

# =====================================================================
# MODELS
# =====================================================================
class BaselineLSTM(nn.Module):
    """Baseline LSTM with optional per-gate LayerNorm (before activation)."""
    def __init__(self, vocab_size, embed_size, hidden_size, use_ln=False):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.W = nn.Linear(embed_size + hidden_size, 4 * hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.hidden_size = hidden_size
        self.use_ln = use_ln
        if use_ln:
            self.ln_i = nn.LayerNorm(hidden_size)
            self.ln_f = nn.LayerNorm(hidden_size)
            self.ln_o = nn.LayerNorm(hidden_size)
            self.ln_g = nn.LayerNorm(hidden_size)

    def forward(self, idx):
        h = torch.zeros(idx.size(0), self.hidden_size, device=idx.device)
        for t in range(idx.size(1)):
            x = self.embed(idx[:, t])
            gates = self.W(torch.cat([x, h], dim=1))
            i, f, o, g = gates.chunk(4, 1)
            if self.use_ln:
                i = self.ln_i(i); f = self.ln_f(f); o = self.ln_o(o); g = self.ln_g(g)
            i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
            g = torch.tanh(g)
            c = f * h + i * g; h = o * torch.tanh(c)
        return self.fc(h).unsqueeze(0)


class HyperLSTMCell(nn.Module):
    """HyperLSTM Cell: per-gate LN (Eq.10) + weight scaling (Eq.11-12) + recurrent dropout (Eq.13)."""
    def __init__(self, input_size, hidden_size, hyper_hidden_size=16, hyper_embedding_size=4,
                 use_ln=False, dropout_prob=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.hyper_hidden_size = hyper_hidden_size
        self.use_ln = use_ln

        self.W_x = nn.Parameter(torch.Tensor(input_size, 4 * hidden_size))
        self.W_h = nn.Parameter(torch.Tensor(hidden_size, 4 * hidden_size))
        self.b   = nn.Parameter(torch.Tensor(4 * hidden_size))

        if use_ln:
            self.ln_i = nn.LayerNorm(hidden_size)
            self.ln_f = nn.LayerNorm(hidden_size)
            self.ln_o = nn.LayerNorm(hidden_size)
            self.ln_g = nn.LayerNorm(hidden_size)

        self.hyper_rnn = nn.LSTMCell(input_size + hidden_size, hyper_hidden_size)

        self.proj_zx = nn.Linear(hyper_hidden_size, hyper_embedding_size)
        self.proj_zh = nn.Linear(hyper_hidden_size, hyper_embedding_size)
        self.proj_zb = nn.Linear(hyper_hidden_size, hyper_embedding_size)

        Nz = hyper_embedding_size
        self.proj_dx = nn.Linear(Nz, 4 * hidden_size)
        self.proj_dh = nn.Linear(Nz, 4 * hidden_size)
        self.proj_db = nn.Linear(Nz, 4 * hidden_size)

        if dropout_prob > 0:
            self.dropout = nn.Dropout(p=dropout_prob)
        self.dropout_prob = dropout_prob

        # Paper initialization
        nn.init.orthogonal_(self.W_x)
        nn.init.orthogonal_(self.W_h)
        nn.init.zeros_(self.b)
        nn.init.zeros_(self.proj_zx.weight); nn.init.ones_(self.proj_zx.bias)
        nn.init.zeros_(self.proj_zh.weight); nn.init.ones_(self.proj_zh.bias)
        nn.init.zeros_(self.proj_zb.weight)
        nn.init.normal_(self.proj_zb.bias, std=0.01)
        nn.init.constant_(self.proj_dx.weight, 0.1 / Nz)
        nn.init.constant_(self.proj_dh.weight, 0.1 / Nz)
        nn.init.constant_(self.proj_db.weight, 0.1 / Nz)

    def forward(self, x_t, state, hyper_state):
        h_prev, c_prev = state
        h_hat_prev, c_hat_prev = hyper_state

        hyper_input = torch.cat([x_t, h_prev], dim=-1)
        h_hat_t, c_hat_t = self.hyper_rnn(hyper_input, (h_hat_prev, c_hat_prev))

        z_x = self.proj_zx(h_hat_t)
        z_h = self.proj_zh(h_hat_t)
        z_b = self.proj_zb(h_hat_t)

        d_x = self.proj_dx(z_x)
        d_h = self.proj_dh(z_h)
        d_b = self.proj_db(z_b)

        gates = (x_t @ self.W_x) * d_x + (h_prev @ self.W_h) * d_h + self.b + d_b

        n = self.b.shape[0]
        i = gates[:, :n//4]
        f = gates[:, n//4:n//2]
        o = gates[:, n//2:3*n//4]
        g = gates[:, 3*n//4:]

        if self.use_ln:
            i = self.ln_i(i); f = self.ln_f(f); o = self.ln_o(o); g = self.ln_g(g)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        if self.dropout_prob > 0 and self.training:
            g = self.dropout(g)

        c_t = f * c_prev + i * g
        h_t = o * torch.tanh(c_t)

        return (h_t, c_t), (h_hat_t, c_hat_t)


class HyperLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size,
                 hyper_hidden_size=16, hyper_embedding_size=4,
                 use_ln=False, dropout_prob=0.0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.hidden_size = hidden_size
        self.cell = HyperLSTMCell(
            embed_size, hidden_size,
            hyper_hidden_size, hyper_embedding_size,
            use_ln, dropout_prob
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, idx):
        h = torch.zeros(idx.size(0), self.hidden_size, device=idx.device)
        c = torch.zeros(idx.size(0), self.hidden_size, device=idx.device)
        hh = torch.zeros(idx.size(0), self.cell.hyper_hidden_size, device=idx.device)
        ch = hh.clone()
        for t in range(idx.size(1)):
            x = self.embed(idx[:, t])
            (h, c), (hh, ch) = self.cell(x, (h, c), (hh, ch))
        return self.fc(h)


# =====================================================================
# DATA
# =====================================================================
DATA_FILE = "shakespeare.txt"
if not os.path.exists(DATA_FILE):
    log("Downloading Shakespeare...")
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        DATA_FILE)

with open(DATA_FILE, 'r', encoding='utf-8') as f:
    text = f.read()

n = len(text)
train_text = text[:int(n*0.9)]
val_text   = text[int(n*0.9):int(n*0.95)]
test_text  = text[int(n*0.95):]

chars = sorted(set(train_text + val_text + test_text))
VS = len(chars)
c2i = {c: i for i, c in enumerate(chars)}
i2c = [c for c in sorted(c2i.keys())]
train_data = torch.tensor([c2i[c] for c in train_text], dtype=torch.long)
val_data   = torch.tensor([c2i[c] for c in val_text],   dtype=torch.long)
test_data  = torch.tensor([c2i[c] for c in test_text],  dtype=torch.long)
log(f"Shakespeare: vocab={VS}, train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")


# =====================================================================
# HELPERS
# =====================================================================
def get_batch(data, seq_len, batch_size, device):
    ix = torch.randint(len(data) - seq_len, (batch_size,))
    x = torch.stack([data[i:i+seq_len] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+seq_len+1] for i in ix]).to(device)
    return x, y

def run_fwd(model, x, device):
    """Run model on sequence, return logits for last timestep. Shape: (batch, vocab)."""
    if isinstance(model, BaselineLSTM):
        h = torch.zeros(x.size(0), model.hidden_size, device=device)
        for t in range(x.size(1)):
            emb = model.embed(x[:, t])
            gates = model.W(torch.cat([emb, h], dim=1))
            i, f, o, g = gates.chunk(4, 1)
            if model.use_ln:
                i = model.ln_i(i); f = model.ln_f(f); o = model.ln_o(o); g = model.ln_g(g)
            i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o); g = torch.tanh(g)
            h = o * torch.tanh(f * h + i * g)
        return model.fc(h)
    # HyperLSTM: forward returns (batch, vocab) directly
    return model(x)

def eval_model(model, data, batch_size, seq_len, device, num_batches=50):
    model.eval()
    crit = nn.CrossEntropyLoss()
    total = 0.0
    with torch.no_grad():
        for _ in range(num_batches):
            x, y = get_batch(data, seq_len, batch_size, device)
            total += crit(run_fwd(model, x, device), y[:, -1]).item()
    model.train()
    return total / num_batches

def train_model(name, model, steps, batch_size, seq_len, device):
    log(f"\nTraining {name}...")
    model.to(device)
    model.train()

    opt = optim.Adam(model.parameters(), lr=0.001)
    crit = nn.CrossEntropyLoss()

    ckpt = os.path.join(OUT, f"{name}.ckpt")
    best = float('inf')
    start = 1

    if os.path.exists(ckpt):
        c = torch.load(ckpt, weights_only=False)
        model.load_state_dict(c['m'])
        opt.load_state_dict(c['o'])
        start = c['s'] + 1
        best = c['b']
        log(f"  Resumed from step {start}")

    t_last = time.time()

    CSV = os.path.join(OUT, "bpc_history.csv")
    csv_f = open(CSV, "a", newline="")
    cw = csv.writer(csv_f)
    if os.path.getsize(CSV) == 0:
        cw.writerow(["Model", "Step", "Val_BPC"])

    for step in range(start, steps + 1):
        x, y = get_batch(train_data, seq_len, batch_size, device)
        opt.zero_grad()
        loss = crit(run_fwd(model, x, device), y[:, -1])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 500 == 0 or step == 1:
            vl = eval_model(model, val_data, batch_size, seq_len, device)
            vbpc = vl / math.log(2)
            elapsed = time.time() - t_last
            speed = 500 / elapsed if step > 1 else 0
            eta = str(datetime.timedelta(seconds=int((steps - step) / speed))) if speed > 0 else "..."
            log(f"  [{name}] {step}/{steps} | Val BPC={vbpc:.4f} | {speed:.1f} it/s | ETA={eta}")
            t_last = time.time()
            cw.writerow([name, step, f"{vbpc:.4f}"])
            csv_f.flush()

            torch.save({
                's': step,
                'm': model.state_dict(),
                'o': opt.state_dict(),
                'b': best
            }, ckpt)

            if vl < best:
                best = vl
                torch.save(model.state_dict(), os.path.join(OUT, f"{name}_best.pth"))
                log(f"    -> Best! BPC={vbpc:.4f}")

    csv_f.close()
    log(f"  [{name}] Training done. Best Val BPC={best/math.log(2):.4f}")
    return best / math.log(2)


# =====================================================================
# MAIN
# =====================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    E, H = 64, 1000
    BS, SL, ST = 128, 100, 20000

    log("=" * 60)
    log(f"Shakespeare (Novel Dataset): embed={E}, hidden={H}, batch={BS}, seq={SL}, steps={ST}")
    log("=" * 60)

    models = [
        ("Baseline_LSTM",         BaselineLSTM(VS, E, H, use_ln=False)),
        ("LN_LSTM",               BaselineLSTM(VS, E, H, use_ln=True)),
        ("Hyper_LSTM",            HyperLSTM(VS, E, H, 16, 4, use_ln=False, dropout_prob=0.0)),
        ("LN_Hyper_LSTM",         HyperLSTM(VS, E, H, 16, 4, use_ln=True, dropout_prob=0.1)),
    ]

    results = {}
    for name, model in models:
        results[name] = train_model(name, model, ST, BS, SL, device)

    log("\n" + "=" * 60)
    log("TRAINING COMPLETE")
    log("=" * 60)
    for n, vbpc in results.items():
        log(f"  {n}: Best Val BPC={vbpc:.4f}")
    log("\nRun eval_shakespeare.py to evaluate on test set and generate samples.")
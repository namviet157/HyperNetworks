"""
Shakespeare: Train only 2 HyperLSTM models with HH=32 (paper default)
Reuses baseline checkpoints from train_shakespeare_results
Run:  CUDA_VISIBLE_DEVICES=7 python3 train_shakespeare_hyper.py
"""

import os, time, datetime, math, csv, shutil, urllib.request, torch, torch.nn as nn, torch.optim as optim

OUT = "train_shakespeare_results"
os.makedirs(OUT, exist_ok=True)
LOG = os.path.join(OUT, "log.txt")

def log(msg):
    print(msg)
    with open(LOG, "a") as f: f.write(msg + "\n")

# =====================================================================
# BASELINE LSTM (paper Eq.9)
# =====================================================================
class BaselineLSTM(nn.Module):
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
                i = self.ln_i(i); f = self.ln_f(f)
                o = self.ln_o(o); g = self.ln_g(g)
            i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
            g = torch.tanh(g)
            h = o * torch.tanh(f * h + i * g)
        return self.fc(h)


# =====================================================================
# HYPERLSTM (paper Eq.10-13, A.2.3) — per-gate separate projections
# =====================================================================
class HyperLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size,
                 hyper_hidden_size=32, hyper_embedding_size=4,
                 use_ln=False, dropout_prob=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.hyper_hidden_size = hyper_hidden_size
        self.hyper_embedding_size = hyper_embedding_size
        self.use_ln = use_ln
        self.dropout_prob = dropout_prob

        # Paper A.2.2: separate weight matrix per gate
        self.Wx_i = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.Wh_i = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.Wx_f = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.Wh_f = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.Wx_g = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.Wh_g = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.Wx_o = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.Wh_o = nn.Parameter(torch.Tensor(hidden_size, hidden_size))

        self.b_i = nn.Parameter(torch.zeros(hidden_size))
        self.b_f = nn.Parameter(torch.zeros(hidden_size))
        self.b_g = nn.Parameter(torch.zeros(hidden_size))
        self.b_o = nn.Parameter(torch.zeros(hidden_size))

        # LayerNorm per gate (paper Eq.10)
        if use_ln:
            self.ln_i = nn.LayerNorm(hidden_size)
            self.ln_f = nn.LayerNorm(hidden_size)
            self.ln_g = nn.LayerNorm(hidden_size)
            self.ln_o = nn.LayerNorm(hidden_size)

        # Hypernetwork (paper Eq.10)
        self.hyper_rnn = nn.LSTMCell(input_size + hidden_size, hyper_hidden_size)

        # Paper A.2.3: per-gate projections
        Nz = hyper_embedding_size
        self.proj_z_i = nn.Linear(hyper_hidden_size, Nz)
        self.proj_z_f = nn.Linear(hyper_hidden_size, Nz)
        self.proj_z_g = nn.Linear(hyper_hidden_size, Nz)
        self.proj_z_o = nn.Linear(hyper_hidden_size, Nz)

        self.proj_d_i = nn.Linear(Nz, hidden_size)
        self.proj_d_f = nn.Linear(Nz, hidden_size)
        self.proj_d_g = nn.Linear(Nz, hidden_size)
        self.proj_d_o = nn.Linear(Nz, hidden_size)

        if dropout_prob > 0:
            self.dropout = nn.Dropout(p=dropout_prob)

        self._init_weights()

    def _init_weights(self):
        Nz = self.hyper_embedding_size
        # Orthogonal for main LSTM weights
        nn.init.orthogonal_(self.Wx_i); nn.init.orthogonal_(self.Wh_i)
        nn.init.orthogonal_(self.Wx_f); nn.init.orthogonal_(self.Wh_f)
        nn.init.orthogonal_(self.Wx_g); nn.init.orthogonal_(self.Wh_g)
        nn.init.orthogonal_(self.Wx_o); nn.init.orthogonal_(self.Wh_o)
        # proj_z: weight=0, bias=1 (z starts near 1)
        nn.init.zeros_(self.proj_z_i.weight); nn.init.ones_(self.proj_z_i.bias)
        nn.init.zeros_(self.proj_z_f.weight); nn.init.ones_(self.proj_z_f.bias)
        nn.init.zeros_(self.proj_z_g.weight); nn.init.ones_(self.proj_z_g.bias)
        nn.init.zeros_(self.proj_z_o.weight); nn.init.ones_(self.proj_z_o.bias)
        # proj_d: 0.1/Nz (d starts near 0.1)
        nn.init.constant_(self.proj_d_i.weight, 0.1 / Nz)
        nn.init.constant_(self.proj_d_f.weight, 0.1 / Nz)
        nn.init.constant_(self.proj_d_g.weight, 0.1 / Nz)
        nn.init.constant_(self.proj_d_o.weight, 0.1 / Nz)

    def forward(self, x_t, state, hyper_state):
        h_prev, c_prev = state
        h_hat_prev, c_hat_prev = hyper_state

        # Hyper network (paper Eq.10)
        hyper_input = torch.cat([x_t, h_prev], dim=-1)
        h_hat_t, c_hat_t = self.hyper_rnn(hyper_input, (h_hat_prev, c_hat_prev))

        # Per-gate scaling (paper Eq.11)
        z_i = self.proj_z_i(h_hat_t); z_f = self.proj_z_f(h_hat_t)
        z_g = self.proj_z_g(h_hat_t); z_o = self.proj_z_o(h_hat_t)

        d_i = self.proj_d_i(z_i); d_f = self.proj_d_f(z_f)
        d_g = self.proj_d_g(z_g); d_o = self.proj_d_o(z_o)

        # Gates with dynamic scaling (paper Eq.12)
        i = (x_t @ self.Wx_i + d_i * (h_prev @ self.Wh_i)) + self.b_i
        f = (x_t @ self.Wx_f + d_f * (h_prev @ self.Wh_f)) + self.b_f
        g = (x_t @ self.Wx_g + d_g * (h_prev @ self.Wh_g)) + self.b_g
        o = (x_t @ self.Wx_o + d_o * (h_prev @ self.Wh_o)) + self.b_o

        if self.use_ln:
            i = self.ln_i(i); f = self.ln_f(f)
            g = self.ln_g(g); o = self.ln_o(o)

        i = torch.sigmoid(i); f = torch.sigmoid(f)
        g = torch.tanh(g); o = torch.sigmoid(o)

        # Recurrent dropout on g (paper Eq.13)
        if self.dropout_prob > 0 and self.training:
            g = self.dropout(g)

        c_t = f * c_prev + i * g
        h_t = o * torch.tanh(c_t)

        return (h_t, c_t), (h_hat_t, c_hat_t)


class HyperLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size,
                 hyper_hidden_size=32, hyper_embedding_size=4,
                 use_ln=False, dropout_prob=0.1):
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
        batch = idx.size(0)
        h = torch.zeros(batch, self.hidden_size, device=idx.device)
        c = torch.zeros(batch, self.hidden_size, device=idx.device)
        hh = torch.zeros(batch, self.cell.hyper_hidden_size, device=idx.device)
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
    if isinstance(model, BaselineLSTM):
        h = torch.zeros(x.size(0), model.hidden_size, device=device)
        for t in range(x.size(1)):
            emb = model.embed(x[:, t])
            gates = model.W(torch.cat([emb, h], dim=1))
            i, f, o, g = gates.chunk(4, 1)
            if model.use_ln:
                i = model.ln_i(i); f = model.ln_f(f)
                o = model.ln_o(o); g = model.ln_g(g)
            i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
            g = torch.tanh(g)
            h = o * torch.tanh(f * h + i * g)
        return model.fc(h)
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

    CSV = os.path.join(OUT, "bpc_history.csv")
    csv_f = open(CSV, "a", newline="")
    cw = csv.writer(csv_f)
    if os.path.getsize(CSV) == 0:
        cw.writerow(["Model", "Step", "Val_BPC"])

    t_last = time.time()

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

            torch.save({'s': step, 'm': model.state_dict(), 'o': opt.state_dict(), 'b': best}, ckpt)

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
    HH, HE = 32, 4    # paper defaults
    KP = 0.9
    BS, SL, ST = 128, 100, 20000

    log("=" * 60)
    log(f"Shakespeare: Train HyperLSTM HH=32 (paper default)")
    log(f"  embed={E}, hidden={H}, hyper_hid={HH}, hyper_emb={HE}")
    log(f"  batch={BS}, seq={SL}, steps={ST}, keep_prob={KP}")
    log("=" * 60)

    models = [
        ("Hyper_LSTM",
         HyperLSTM(VS, E, H, HH, HE, use_ln=False, dropout_prob=1-KP)),
        ("LN_Hyper_LSTM",
         HyperLSTM(VS, E, H, HH, HE, use_ln=True,  dropout_prob=1-KP)),
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

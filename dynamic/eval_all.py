"""
Unified Evaluation: PTB + enwik8 + Shakespeare
Loads checkpoints from each dataset folder, evaluates on test set,
and saves results + samples to CSV.

Run:  CUDA_VISIBLE_DEVICES=0 python eval_all.py

Folders expected:
  results_ptb_final/          -> PTB checkpoints
  results_enwik8_final/       -> enwik8 checkpoints
  train_shakespeare_results/   -> Shakespeare checkpoints
"""

import os, math, csv, time, urllib.request, torch, torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================================================================
# PTB / enwik8 models (CharLM wrapper, shared architecture)
# =====================================================================
class BaselineLSTM_PTB(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_keep=0.9, use_ln=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_ln = use_ln
        self.dropout = nn.Dropout(1 - dropout_keep) if dropout_keep < 1.0 else nn.Identity()
        self.W = nn.Linear(input_size + hidden_size, 4 * hidden_size)
        if use_ln:
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
        for t in range(seq_len):
            gates = self.W(torch.cat([x[:, t, :], h], dim=1))
            i, f, o, g = gates.chunk(4, 1)
            if self.use_ln:
                i = self.ln_i(i); f = self.ln_f(f)
                o = self.ln_o(o); g = self.ln_g(g)
            i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
            g = torch.tanh(g)
            c = f * c + i * self.dropout(g)
            c_norm = self.ln_c(c) if self.use_ln else c
            h = o * torch.tanh(c_norm)
        return h


class HyperLSTM_PTB(nn.Module):
    def __init__(self, input_size, hidden_size, hyper_hidden_size=128,
                 hyper_embed_size=4, dropout_keep=0.9, use_ln=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.hyper_hidden_size = hyper_hidden_size
        self.hyper_embed_size = hyper_embed_size
        self.use_ln = use_ln
        self.dropout = nn.Dropout(1 - dropout_keep) if dropout_keep < 1.0 else nn.Identity()
        self.W = nn.Linear(input_size + hidden_size, 4 * hidden_size, bias=False)
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
        self.proj_z_x = nn.Linear(hyper_hidden_size, 4 * hyper_embed_size)
        self.proj_z_h = nn.Linear(hyper_hidden_size, 4 * hyper_embed_size)
        self.proj_z_b = nn.Linear(hyper_hidden_size, 4 * hyper_embed_size)
        self.proj_d_x = nn.Linear(hyper_embed_size, hidden_size, bias=False)
        self.proj_d_h = nn.Linear(hyper_embed_size, hidden_size, bias=False)
        self.proj_bias = nn.Linear(hyper_embed_size, hidden_size)
        if use_ln:
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
        for t in range(seq_len):
            x_t = x[:, t, :]
            i_t = self.ln_i(self.Wi_h(h_hat) + self.Wi_x(x_t) + self.bi)
            g_t = self.ln_g(self.Wg_h(h_hat) + self.Wg_x(x_t) + self.bg)
            f_t = self.ln_f(self.Wf_h(h_hat) + self.Wf_x(x_t) + self.bf)
            o_t = self.ln_o(self.Wo_h(h_hat) + self.Wo_x(x_t) + self.bo)
            i_t = torch.sigmoid(i_t); g_t = torch.tanh(g_t)
            f_t = torch.sigmoid(f_t); o_t = torch.sigmoid(o_t)
            c_hat = f_t * c_hat + i_t * g_t
            h_hat = o_t * torch.tanh(self.ln_c(c_hat))
            zx = self.proj_z_x(h_hat)
            zh = self.proj_z_h(h_hat)
            zb = self.proj_z_b(h_hat)
            dx = torch.cat([self.proj_d_x(z) for z in zx.chunk(4, -1)], dim=-1)
            dh = torch.cat([self.proj_d_h(z) for z in zh.chunk(4, -1)], dim=-1)
            bd = torch.cat([self.proj_bias(z) for z in zb.chunk(4, -1)], dim=-1)
            W_x = nn.functional.linear(x_t, self.W.weight[:, :x_t.shape[-1]])
            W_h = nn.functional.linear(h,     self.W.weight[:, x_t.shape[-1]:])
            gates = W_x * dx + W_h * dh + bd
            i, f, o, g = gates.chunk(4, -1)
            if self.use_ln:
                i = self.ln_i_main(i); f = self.ln_f_main(f)
                o = self.ln_o_main(o); g = self.ln_g_main(g)
            i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
            g = torch.tanh(g)
            c = f * c + i * self.dropout(g)
            c_norm = self.ln_c_main(c) if self.use_ln else c
            h = o * torch.tanh(c_norm)
        return h


class CharLM(nn.Module):
    def __init__(self, core, vocab_size, embed_size, hidden_size, dropout_keep=0.9):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.drop  = nn.Dropout(1 - dropout_keep)
        self.core  = core
        self.out   = nn.Linear(hidden_size, vocab_size)

    def forward(self, idx):
        x = self.drop(self.embed(idx))
        h = self.core(x)
        return self.out(h)


# =====================================================================
# Shakespeare models (direct, no CharLM wrapper)
# =====================================================================
# Shakespeare models — exact architecture from train_shakespeare_hyper.py
# =====================================================================
class BaselineLSTM_Shk(nn.Module):
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


class HyperLSTMCell(nn.Module):
    """Per-gate separate weights (paper A.2.2) — matches train_shakespeare_hyper.py."""
    def __init__(self, input_size, hidden_size,
                 hyper_hidden_size=32, hyper_embedding_size=4,
                 use_ln=False, dropout_prob=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.hyper_hidden_size = hyper_hidden_size
        self.use_ln = use_ln
        self.dropout_prob = dropout_prob
        Nz = hyper_embedding_size

        # Per-gate weight matrices (paper A.2.2)
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

        if use_ln:
            self.ln_i = nn.LayerNorm(hidden_size)
            self.ln_f = nn.LayerNorm(hidden_size)
            self.ln_g = nn.LayerNorm(hidden_size)
            self.ln_o = nn.LayerNorm(hidden_size)

        self.hyper_rnn = nn.LSTMCell(input_size + hidden_size, hyper_hidden_size)

        # Per-gate projections (paper A.2.3)
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

    def forward(self, x_t, state, hyper_state):
        h_prev, c_prev = state
        h_hat_prev, c_hat_prev = hyper_state
        hyper_input = torch.cat([x_t, h_prev], dim=-1)
        h_hat_t, c_hat_t = self.hyper_rnn(hyper_input, (h_hat_prev, c_hat_prev))
        z_i = self.proj_z_i(h_hat_t); z_f = self.proj_z_f(h_hat_t)
        z_g = self.proj_z_g(h_hat_t); z_o = self.proj_z_o(h_hat_t)
        d_i = self.proj_d_i(z_i); d_f = self.proj_d_f(z_f)
        d_g = self.proj_d_g(z_g); d_o = self.proj_d_o(z_o)
        i = (x_t @ self.Wx_i + d_i * (h_prev @ self.Wh_i)) + self.b_i
        f = (x_t @ self.Wx_f + d_f * (h_prev @ self.Wh_f)) + self.b_f
        g = (x_t @ self.Wx_g + d_g * (h_prev @ self.Wh_g)) + self.b_g
        o = (x_t @ self.Wx_o + d_o * (h_prev @ self.Wh_o)) + self.b_o
        if self.use_ln:
            i = self.ln_i(i); f = self.ln_f(f); g = self.ln_g(g); o = self.ln_o(o)
        i = torch.sigmoid(i); f = torch.sigmoid(f); g = torch.tanh(g); o = torch.sigmoid(o)
        if self.dropout_prob > 0 and self.training:
            g = self.dropout(g)
        c_t = f * c_prev + i * g
        h_t = o * torch.tanh(c_t)
        return (h_t, c_t), (h_hat_t, c_hat_t)


class HyperLSTM_Shk(nn.Module):
    """Full HyperLSTM for Shakespeare — matches train_shakespeare_hyper.py."""
    def __init__(self, vocab_size, embed_size, hidden_size,
                 hyper_hidden_size=32, hyper_embedding_size=4,
                 use_ln=False, dropout_prob=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.hidden_size = hidden_size
        self.hyper_hidden_size = hyper_hidden_size
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
# DATA LOADING
# =====================================================================
def load_ptb():
    train_text = open("ptb.train.txt").read().strip()
    val_text   = open("ptb.valid.txt").read().strip()
    test_text  = open("ptb.test.txt").read().strip()
    chars = sorted(set(train_text + val_text + test_text))
    VS = len(chars)
    c2i = {c: i for i, c in enumerate(chars)}
    i2c = list(chars)
    train_data = torch.tensor([c2i[c] for c in train_text], dtype=torch.long)
    val_data   = torch.tensor([c2i[c] for c in val_text],   dtype=torch.long)
    test_data  = torch.tensor([c2i[c] for c in test_text],  dtype=torch.long)
    return train_data, val_data, test_data, VS, i2c, c2i


def load_shakespeare():
    DATA_FILE = "shakespeare.txt"
    if not os.path.exists(DATA_FILE):
        print("  Downloading Shakespeare data...")
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
            DATA_FILE)
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        text = f.read()
    n = len(text)
    train_text = text[:int(n*0.9)]
    val_text   = text[int(n*0.9):int(n*0.95)]
    test_text  = text[int(n*0.95):]
    chars = sorted(set(text))
    VS = len(chars)
    c2i = {c: i for i, c in enumerate(chars)}
    i2c = list(chars)
    train_data = torch.tensor([c2i[c] for c in train_text], dtype=torch.long)
    val_data   = torch.tensor([c2i[c] for c in val_text],   dtype=torch.long)
    test_data  = torch.tensor([c2i[c] for c in test_text],  dtype=torch.long)
    return train_data, val_data, test_data, VS, i2c, c2i


def load_enwik8():
    DATA_FILE = "enwik8"
    if not os.path.exists(DATA_FILE):
        print("  Downloading enwik8 data...")
        import zipfile
        urllib.request.urlretrieve("https://data.deepai.org/enwik8.zip", "enwik8.zip")
        with zipfile.ZipFile("enwik8.zip", 'r') as z:
            z.extractall()
        os.remove("enwik8.zip")
    with open(DATA_FILE, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    train_text = text[:90000000]
    val_text   = text[90000000:95000000]
    test_text  = text[95000000:]
    chars = sorted(set(text))
    VS = len(chars)
    c2i = {c: i for i, c in enumerate(chars)}
    i2c = list(chars)
    train_data = torch.tensor([c2i[c] for c in train_text], dtype=torch.long)
    val_data   = torch.tensor([c2i[c] for c in val_text],   dtype=torch.long)
    test_data  = torch.tensor([c2i[c] for c in test_text],  dtype=torch.long)
    return train_data, val_data, test_data, VS, i2c, c2i


# =====================================================================
# EVALUATION HELPERS
# =====================================================================
def get_batch(data, seq_len, batch_size, device):
    ix = torch.randint(len(data) - seq_len, (batch_size,))
    x = torch.stack([data[i:i+seq_len] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+seq_len+1] for i in ix]).to(device)
    return x, y


def eval_model(model, data, batch_size, seq_len, device, num_batches=200):
    model.eval()
    crit = nn.CrossEntropyLoss()
    total = 0.0
    with torch.no_grad():
        for _ in range(num_batches):
            x, y = get_batch(data, seq_len, batch_size, device)
            logits = model(x)
            total += crit(logits, y[:, -1]).item()
    model.train()
    return total / num_batches


def _sample_step_charlm(core, model, idx, h, c, device):
    """One step for CharLM (PTB/enwik8). Returns (h, c, logits)."""
    x_t = model.drop(model.embed(idx))
    # Run core step-by-step
    gates = core.W(torch.cat([x_t.squeeze(1), h], dim=1))
    i, f, o, g = gates.chunk(4, 1)
    if core.use_ln:
        i = core.ln_i(i); f = core.ln_f(f); o = core.ln_o(o); g = core.ln_g(g)
    i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
    g = torch.tanh(g)
    c = f * c + i * core.dropout(g)
    c_norm = core.ln_c(c) if core.use_ln else c
    h = o * torch.tanh(c_norm)
    logits = model.out(model.drop(h))
    return h, c, logits


def _sample_step_baseline_shk(core, idx, h, c):
    x_t = core.embed(idx)
    gates = core.W(torch.cat([x_t.squeeze(1), h], dim=1))
    i, f, o, g = gates.chunk(4, 1)
    if core.use_ln:
        i = core.ln_i(i); f = core.ln_f(f); o = core.ln_o(o); g = core.ln_g(g)
    i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
    g = torch.tanh(g)
    c = f * c + i * g
    h = o * torch.tanh(c)
    logits = core.fc(h)
    return h, c, logits


def _sample_step_hyper_shk(core, idx, h, c, hh, ch):
    """Per-gate architecture matching HyperLSTMCell."""
    x_t = core.embed(idx)
    hyper_input = torch.cat([x_t.squeeze(1), h], dim=-1)
    hh, ch = core.hyper_rnn(hyper_input, (hh, ch))
    z_i = core.proj_z_i(hh); z_f = core.proj_z_f(hh)
    z_g = core.proj_z_g(hh); z_o = core.proj_z_o(hh)
    d_i = core.proj_d_i(z_i); d_f = core.proj_d_f(z_f)
    d_g = core.proj_d_g(z_g); d_o = core.proj_d_o(z_o)
    i = (x_t.squeeze(1) @ core.Wx_i + d_i * (h @ core.Wh_i)) + core.b_i
    f = (x_t.squeeze(1) @ core.Wx_f + d_f * (h @ core.Wh_f)) + core.b_f
    g = (x_t.squeeze(1) @ core.Wx_g + d_g * (h @ core.Wh_g)) + core.b_g
    o = (x_t.squeeze(1) @ core.Wx_o + d_o * (h @ core.Wh_o)) + core.b_o
    if core.use_ln:
        i = core.ln_i(i); f = core.ln_f(f); g = core.ln_g(g); o = core.ln_o(o)
    i = torch.sigmoid(i); f = torch.sigmoid(f); g = torch.tanh(g); o = torch.sigmoid(o)
    if core.dropout_prob > 0 and core.training:
        g = core.dropout(g)
    c = f * c + i * g
    h = o * torch.tanh(c)
    logits = core.fc(h)
    return h, c, logits, hh, ch


def sample_text(model, i2c, c2i, device, seed="ROMEO:", length=100):
    model.eval()
    chars = list(seed)

    is_charlm = isinstance(model, CharLM)
    core = model.core if is_charlm else model
    hidden_size = core.hidden_size

    is_baseline_shk = isinstance(core, BaselineLSTM_Shk)
    is_hyper_shk = isinstance(core, HyperLSTM_Shk)

    h = torch.zeros(1, hidden_size, device=device)
    c = torch.zeros(1, hidden_size, device=device)
    hh = torch.zeros(1, core.hyper_hidden_size) if is_hyper_shk else None
    ch = hh.clone() if hh is not None else None

    with torch.no_grad():
        # Prime
        for ch_input in chars:
            idx = torch.tensor([[c2i.get(ch_input, 0)]], device=device)
            if is_charlm:
                h, c, logits = _sample_step_charlm(core, model, idx, h, c, device)
            elif is_baseline_shk:
                h, c, logits = _sample_step_baseline_shk(core, idx, h, c)
            elif is_hyper_shk:
                h, c, logits, hh, ch = _sample_step_hyper_shk(core, idx, h, c, hh, ch)
            probs = torch.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, 1).item()
            chars.append(i2c[next_idx] if next_idx < len(i2c) else "")

        # Generate
        for _ in range(length):
            idx = torch.tensor([[c2i.get(chars[-1], 0)]], device=device)
            if is_charlm:
                h, c, logits = _sample_step_charlm(core, model, idx, h, c, device)
            elif is_baseline_shk:
                h, c, logits = _sample_step_baseline_shk(core, idx, h, c)
            elif is_hyper_shk:
                h, c, logits, hh, ch = _sample_step_hyper_shk(core, idx, h, c, hh, ch)
            probs = torch.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, 1).item()
            chars.append(i2c[next_idx] if next_idx < len(i2c) else "")

    model.train()
    return "".join(chars)


# =====================================================================
# DATASET CONFIGS
# =====================================================================
CONFIGS = {
    "ptb": {
        "load_fn": load_ptb,
        "out_dir": "results_ptb_final",
        "log_file": "progress_log.txt",
        "embed": 64, "hidden": 1000, "hh": 128, "he": 4, "kp": 0.9,
        "batch": 128, "seq": 100,
        "models": [
            ("LSTM_Baseline_1000",          "BaselineLSTM",  {}),
            ("LayerNorm_LSTM_1000",         "BaselineLSTM",  {"use_ln": True}),
            ("HyperLSTM_1000",              "HyperLSTM",     {}),
            ("LayerNorm_HyperLSTM_1000",    "HyperLSTM",     {"use_ln": True}),
        ],
    },
    "enwik8": {
        "load_fn": load_enwik8,
        "out_dir": "results_enwik8_final",
        "log_file": "progress_log.txt",
        "embed": 64, "hidden": 1000, "hh": 128, "he": 64, "kp": 0.9,
        "batch": 64, "seq": 250,
        "models": [
            ("LSTM_no_dropout",            "BaselineLSTM",  {}),
            ("LayerNorm_LSTM",             "BaselineLSTM",  {"use_ln": True}),
            ("HyperLSTM",                  "HyperLSTM",    {}),
            ("LayerNorm_HyperLSTM",        "HyperLSTM",    {"use_ln": True}),
        ],
    },
    "shakespeare": {
        "load_fn": load_shakespeare,
        "out_dir": "train_shakespeare_results",
        "log_file": "log.txt",
        "embed": 64, "hidden": 1000, "hh": 32, "he": 4, "kp": 0.9,
        "batch": 128, "seq": 100,
        "models": [
            ("Baseline_LSTM",              "BaselineLSTM_Shk", {}),
            ("LN_LSTM",                   "BaselineLSTM_Shk", {"use_ln": True}),
            ("Hyper_LSTM",                "HyperLSTM_Shk",   {"dropout_prob": 0.1}),
            ("LN_Hyper_LSTM",             "HyperLSTM_Shk",   {"use_ln": True, "dropout_prob": 0.1}),
        ],
    },
}


# =====================================================================
# MAIN
# =====================================================================
if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    os.makedirs("eval_results", exist_ok=True)
    csv_path = os.path.join("eval_results", "summary.csv")
    csv_file = open(csv_path, "a", newline="")
    cw = csv.writer(csv_file)
    if os.path.getsize(csv_path) == 0:
        cw.writerow(["Dataset", "Model", "Val_BPC", "Test_BPC", "Checkpoint"])

    for ds_name, cfg in CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name.upper()}")
        print(f"{'='*60}")

        if not os.path.exists(cfg["out_dir"]):
            print(f"  SKIPPED — folder '{cfg['out_dir']}' not found")
            continue

        train_data, val_data, test_data, VS, i2c, c2i = cfg["load_fn"]()
        print(f"  Vocab={VS} | Train={len(train_data):,} | Val={len(val_data):,} | Test={len(test_data):,}")

        for model_name, model_type, extra_kwargs in cfg["models"]:
            ckpt_path = os.path.join(cfg["out_dir"], f"{model_name}_best.pth")
            if not os.path.exists(ckpt_path):
                print(f"  [{model_name}] SKIP — {ckpt_path} not found")
                continue

            # Build model
            if model_type == "BaselineLSTM":
                core = BaselineLSTM_PTB(cfg["embed"], cfg["hidden"], cfg["kp"], **extra_kwargs)
                model = CharLM(core, VS, cfg["embed"], cfg["hidden"], cfg["kp"])
            elif model_type == "HyperLSTM":
                core = HyperLSTM_PTB(cfg["embed"], cfg["hidden"], cfg["hh"], cfg["he"], cfg["kp"], **extra_kwargs)
                model = CharLM(core, VS, cfg["embed"], cfg["hidden"], cfg["kp"])
            elif model_type == "BaselineLSTM_Shk":
                model = BaselineLSTM_Shk(VS, cfg["embed"], cfg["hidden"], **extra_kwargs)
            elif model_type == "HyperLSTM_Shk":
                model = HyperLSTM_Shk(VS, cfg["embed"], cfg["hidden"], cfg["hh"], cfg["he"], **extra_kwargs)

            # Load checkpoint
            state = torch.load(ckpt_path, weights_only=False, map_location=DEVICE)
            model.load_state_dict(state)
            model.to(DEVICE)

            # Evaluate
            vbpc = eval_model(model, val_data,   cfg["batch"], cfg["seq"], DEVICE, num_batches=50)
            tbpc = eval_model(model, test_data,  cfg["batch"], cfg["seq"], DEVICE, num_batches=100)
            print(f"  [{model_name}] Val BPC={vbpc:.4f} | Test BPC={tbpc:.4f}")
            cw.writerow([ds_name, model_name, f"{vbpc:.4f}", f"{tbpc:.4f}", ckpt_path])
            print(f"  [{model_name}] Done. Val BPC={vbpc:.4f}, Test BPC={tbpc:.4f}", flush=True)

            # Sampling — skip for now, run separately
            # seed = "ROMEO:" if ds_name == "shakespeare" else "The "
            # sample = sample_text(model, i2c, c2i, DEVICE, seed=seed, length=50)
            # print(f"  [{model_name}] Sample done: {len(sample)} chars", flush=True)
            # sample_path = os.path.join("eval_results", f"sample_{ds_name}_{model_name}.txt")
            # with open(sample_path, "w") as f: f.write(sample)
            print(f"  [{model_name}] Sample saved.")

            csv_file.flush()

    csv_file.close()

    # =====================================================================
    # ABLATION RESULTS (from eval_ablation_results/)
    # =====================================================================
    print(f"\n{'='*60}")
    print(f"ABLATION STUDY RESULTS")
    print(f"{'='*60}")
    abla_csv = "eval_ablation_results/ablation_summary.csv"
    if os.path.exists(abla_csv):
        print(f"  Loading from {abla_csv}")
        with open(abla_csv, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        # Group by experiment
        exp_groups = {}
        for row in rows:
            exp = row["Experiment"]
            if exp not in exp_groups:
                exp_groups[exp] = []
            exp_groups[exp].append(row)
        for exp, variants in exp_groups.items():
            print(f"\n  [{exp}]")
            for v in variants:
                print(f"    {v['Variant']:20s} Val BPC = {v['Val_BPC']}")
        print(f"\n  Full table saved at: {abla_csv}")
    else:
        print(f"  SKIPPED — {abla_csv} not found")
        print(f"  Run ablation with: python3 eval_ablation.py")

    print(f"\n{'='*60}")
    print(f"ALL DONE! Results saved to eval_results/")
    print(f"{'='*60}")

"""Utilities for loading character-level demo corpora."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple
import urllib.request

import torch
from torch import Tensor

DEFAULT_TEXT = (
    "hypernetworks let a small network tune the weights of a larger network over time.\n"
    "this repository demonstrates a dynamic hypernetwork based on the hyperlstm idea.\n"
    "we train a character-level language model so the behaviour of the model can be shown live.\n"
    "for a stronger demo, run the scripts on tiny shakespeare or on your own report corpus.\n"
) * 64

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


@dataclass
class CharCorpus:
    train_data: Tensor
    val_data: Tensor
    test_data: Tensor
    stoi: Dict[str, int]
    itos: Dict[int, str]
    raw_text: str
    source_name: str

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode_text(self, text: str) -> Tensor:
        token_ids = [self.stoi[char] for char in text if char in self.stoi]
        if not token_ids:
            token_ids = [0]
        return torch.tensor(token_ids, dtype=torch.long)

    def decode_tokens(self, tokens: Tensor) -> str:
        return "".join(self.itos[int(token)] for token in tokens)


def maybe_download_tiny_shakespeare(data_dir: str | Path = "data") -> Path:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / "tinyshakespeare.txt"
    if not target.exists():
        urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, target)
    return target


def load_text(
    *,
    text_file: str | None = None,
    download_tinyshakespeare: bool = False,
    data_dir: str | Path = "data",
) -> Tuple[str, str]:
    if text_file is not None:
        path = Path(text_file)
        if not path.exists():
            raise FileNotFoundError(f"text file not found: {path}")
        return path.read_text(encoding="utf-8"), path.name

    if download_tinyshakespeare:
        path = maybe_download_tiny_shakespeare(data_dir)
        return path.read_text(encoding="utf-8"), path.name

    return DEFAULT_TEXT, "builtin_demo_text"


def build_vocab(text: str) -> Tuple[Dict[str, int], Dict[int, str], Tensor]:
    vocab = sorted(set(text))
    stoi = {char: idx for idx, char in enumerate(vocab)}
    itos = {idx: char for char, idx in stoi.items()}
    encoded = torch.tensor([stoi[char] for char in text], dtype=torch.long)
    return stoi, itos, encoded


def split_encoded_data(
    encoded: Tensor,
    *,
    train_fraction: float = 0.9,
    val_fraction: float = 0.05,
) -> Tuple[Tensor, Tensor, Tensor]:
    if encoded.numel() < 32:
        raise ValueError("corpus is too small; please use a larger text input")

    train_end = int(encoded.numel() * train_fraction)
    val_end = train_end + int(encoded.numel() * val_fraction)

    train_data = encoded[:train_end]
    val_data = encoded[train_end:val_end]
    test_data = encoded[val_end:]

    if val_data.numel() == 0:
        val_data = train_data.clone()
    if test_data.numel() == 0:
        test_data = val_data.clone()

    return train_data, val_data, test_data


def prepare_corpus(
    *,
    text_file: str | None = None,
    download_tinyshakespeare: bool = False,
    data_dir: str | Path = "data",
    train_fraction: float = 0.9,
    val_fraction: float = 0.05,
) -> CharCorpus:
    text, source_name = load_text(
        text_file=text_file,
        download_tinyshakespeare=download_tinyshakespeare,
        data_dir=data_dir,
    )
    stoi, itos, encoded = build_vocab(text)
    train_data, val_data, test_data = split_encoded_data(
        encoded,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
    )
    return CharCorpus(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        stoi=stoi,
        itos=itos,
        raw_text=text,
        source_name=source_name,
    )


def sample_batch(data: Tensor, batch_size: int, seq_len: int, device: torch.device) -> Tuple[Tensor, Tensor]:
    max_start = data.size(0) - seq_len - 1
    if max_start <= 0:
        raise ValueError("corpus is too small for the requested sequence length")

    starts = torch.randint(0, max_start, (batch_size,))
    x_batch = torch.stack([data[start : start + seq_len] for start in starts]).to(device)
    y_batch = torch.stack([data[start + 1 : start + seq_len + 1] for start in starts]).to(device)
    return x_batch, y_batch

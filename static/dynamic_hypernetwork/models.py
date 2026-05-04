"""Character-level language models for the demo."""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor, nn

from .hyperlstm import HyperLSTM


class CharLanguageModelBase(nn.Module):
    def __init__(self, vocab_size: int, embedding_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)

    @torch.no_grad()
    def sample(
        self,
        prompt: str,
        stoi: Dict[str, int],
        itos: Dict[int, str],
        *,
        length: int,
        device: torch.device,
        temperature: float = 0.9,
    ) -> str:
        self.eval()
        prompt = prompt or next(iter(stoi))
        prompt_ids = [stoi[char] for char in prompt if char in stoi]
        if not prompt_ids:
            prompt_ids = [0]

        state = None
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        logits, state = self(input_ids, state)
        generated = [itos[idx] for idx in prompt_ids]
        step_logits = logits[:, -1, :] / temperature

        for _ in range(length):
            probs = torch.softmax(step_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated.append(itos[int(next_token.item())])
            logits, state = self(next_token, state)
            step_logits = logits[:, -1, :] / temperature

        return "".join(generated)


class VanillaCharLSTM(CharLanguageModelBase):
    def __init__(
        self,
        *,
        vocab_size: int,
        embedding_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(vocab_size=vocab_size, embedding_size=embedding_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, tokens: Tensor, state: Optional[tuple[Tensor, Tensor]] = None):
        x = self.embedding(tokens)
        h, state = self.rnn(x, state)
        logits = self.head(h)
        return logits, state


class HyperCharLSTM(CharLanguageModelBase):
    def __init__(
        self,
        *,
        vocab_size: int,
        embedding_size: int,
        hidden_size: int,
        hyper_hidden_size: int,
        hyper_embedding_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        recurrent_dropout: float = 0.0,
    ) -> None:
        super().__init__(vocab_size=vocab_size, embedding_size=embedding_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = HyperLSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            hyper_hidden_size=hyper_hidden_size,
            hyper_embedding_size=hyper_embedding_size,
            num_layers=num_layers,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            batch_first=True,
            use_layer_norm=True,
        )
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, tokens: Tensor, state=None):
        x = self.embedding(tokens)
        h, state = self.rnn(x, state)
        logits = self.head(h)
        return logits, state


def build_model(
    model_name: str,
    *,
    vocab_size: int,
    embedding_size: int,
    hidden_size: int,
    num_layers: int = 1,
    dropout: float = 0.0,
    hyper_hidden_size: int = 32,
    hyper_embedding_size: int = 8,
    recurrent_dropout: float = 0.0,
) -> CharLanguageModelBase:
    model_name = model_name.lower()
    if model_name == "lstm":
        return VanillaCharLSTM(
            vocab_size=vocab_size,
            embedding_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
    if model_name == "hyperlstm":
        return HyperCharLSTM(
            vocab_size=vocab_size,
            embedding_size=embedding_size,
            hidden_size=hidden_size,
            hyper_hidden_size=hyper_hidden_size,
            hyper_embedding_size=hyper_embedding_size,
            num_layers=num_layers,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
        )
    raise ValueError(f"unknown model_name: {model_name}")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

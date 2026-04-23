"""Training and evaluation helpers for the demo."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
from typing import Any, Dict

import torch
from torch import nn
import torch.nn.functional as F

from .data import CharCorpus, sample_batch
from .models import CharLanguageModelBase, build_model, count_parameters


@dataclass
class TrainConfig:
    model_name: str
    device: str = "cpu"
    steps: int = 300
    batch_size: int = 16
    sequence_length: int = 64
    embedding_size: int = 32
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.0
    hyper_hidden_size: int = 32
    hyper_embedding_size: int = 8
    recurrent_dropout: float = 0.0
    learning_rate: float = 3e-3
    grad_clip: float = 1.0
    eval_every: int = 50
    eval_batches: int = 10
    sample_every: int = 100
    sample_length: int = 240
    sample_temperature: float = 0.9
    prompt: str = "To be, or not to be"
    seed: int = 7


def resolve_device(raw_device: str) -> torch.device:
    if raw_device != "auto":
        return torch.device(raw_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tensor_to_float(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def perplexity_from_loss(loss_value: float) -> float:
    return math.exp(loss_value) if loss_value < 20 else float("inf")


def evaluate_model(
    model: CharLanguageModelBase,
    data: torch.Tensor,
    *,
    batch_size: int,
    sequence_length: int,
    batches: int,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(batches):
            x_batch, y_batch = sample_batch(data, batch_size, sequence_length, device)
            logits, _ = model(x_batch)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y_batch.reshape(-1))
            losses.append(tensor_to_float(loss))

    avg_loss = sum(losses) / len(losses)
    return {"loss": avg_loss, "perplexity": perplexity_from_loss(avg_loss)}


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def checkpoint_payload(
    *,
    model: CharLanguageModelBase,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    corpus: CharCorpus,
    step: int,
    val_metrics: Dict[str, float],
) -> Dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": asdict(config),
        "model_kwargs": {
            "vocab_size": corpus.vocab_size,
            "embedding_size": config.embedding_size,
            "hidden_size": config.hidden_size,
            "num_layers": config.num_layers,
            "dropout": config.dropout,
            "hyper_hidden_size": config.hyper_hidden_size,
            "hyper_embedding_size": config.hyper_embedding_size,
            "recurrent_dropout": config.recurrent_dropout,
        },
        "step": step,
        "val_metrics": val_metrics,
        "stoi": corpus.stoi,
        "itos": corpus.itos,
        "source_name": corpus.source_name,
    }


def train_model(
    *,
    corpus: CharCorpus,
    config: TrainConfig,
    output_dir: str | Path,
) -> Dict[str, Any]:
    set_seed(config.seed)
    device = resolve_device(config.device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(
        config.model_name,
        vocab_size=corpus.vocab_size,
        embedding_size=config.embedding_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
        hyper_hidden_size=config.hyper_hidden_size,
        hyper_embedding_size=config.hyper_embedding_size,
        recurrent_dropout=config.recurrent_dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    config_path = output_dir / "config.json"
    history_path = output_dir / "history.jsonl"
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    sample_dir = output_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    save_json(
        config_path,
        {
            "train_config": asdict(config),
            "source_name": corpus.source_name,
            "vocab_size": corpus.vocab_size,
            "num_parameters": count_parameters(model),
        },
    )
    if history_path.exists():
        history_path.unlink()

    best_val_loss = float("inf")
    best_step = 0

    print(
        f"[train] model={config.model_name} device={device} source={corpus.source_name} "
        f"vocab={corpus.vocab_size} params={count_parameters(model)}"
    )

    for step in range(1, config.steps + 1):
        model.train()
        x_batch, y_batch = sample_batch(corpus.train_data, config.batch_size, config.sequence_length, device)
        logits, _ = model(x_batch)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y_batch.reshape(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip)
        optimizer.step()

        if step == 1 or step % config.eval_every == 0 or step == config.steps:
            train_metrics = {"loss": tensor_to_float(loss), "perplexity": perplexity_from_loss(tensor_to_float(loss))}
            val_metrics = evaluate_model(
                model,
                corpus.val_data,
                batch_size=config.batch_size,
                sequence_length=config.sequence_length,
                batches=config.eval_batches,
                device=device,
            )
            record = {
                "step": step,
                "train_loss": train_metrics["loss"],
                "train_perplexity": train_metrics["perplexity"],
                "val_loss": val_metrics["loss"],
                "val_perplexity": val_metrics["perplexity"],
            }
            append_jsonl(history_path, record)
            print(
                f"[eval] step={step:04d} train_loss={record['train_loss']:.4f} "
                f"val_loss={record['val_loss']:.4f} val_ppl={record['val_perplexity']:.2f}"
            )

            checkpoint = checkpoint_payload(
                model=model,
                optimizer=optimizer,
                config=config,
                corpus=corpus,
                step=step,
                val_metrics=val_metrics,
            )
            torch.save(checkpoint, last_path)

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_step = step
                torch.save(checkpoint, best_path)

        if step == 1 or step % config.sample_every == 0 or step == config.steps:
            sample_text = model.sample(
                prompt=config.prompt,
                stoi=corpus.stoi,
                itos=corpus.itos,
                length=config.sample_length,
                device=device,
                temperature=config.sample_temperature,
            )
            sample_path = sample_dir / f"sample_step_{step:04d}.txt"
            sample_path.write_text(sample_text, encoding="utf-8")
            print(f"[sample] saved {sample_path}")

    best_checkpoint = torch.load(best_path, map_location=device)
    best_model = build_model(best_checkpoint["config"]["model_name"], **best_checkpoint["model_kwargs"]).to(device)
    best_model.load_state_dict(best_checkpoint["model_state_dict"])

    test_metrics = evaluate_model(
        best_model,
        corpus.test_data,
        batch_size=config.batch_size,
        sequence_length=config.sequence_length,
        batches=config.eval_batches,
        device=device,
    )

    final_sample = best_model.sample(
        prompt=config.prompt,
        stoi=corpus.stoi,
        itos=corpus.itos,
        length=config.sample_length,
        device=device,
        temperature=config.sample_temperature,
    )
    (output_dir / "final_sample.txt").write_text(final_sample, encoding="utf-8")

    summary = {
        "model_name": config.model_name,
        "source_name": corpus.source_name,
        "device": str(device),
        "num_parameters": count_parameters(best_model),
        "best_step": best_step,
        "best_val_loss": best_val_loss,
        "best_val_perplexity": perplexity_from_loss(best_val_loss),
        "test_loss": test_metrics["loss"],
        "test_perplexity": test_metrics["perplexity"],
        "output_dir": str(output_dir),
    }
    save_json(output_dir / "summary.json", summary)
    return summary


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str = "cpu",
) -> tuple[CharLanguageModelBase, Dict[str, int], Dict[int, str], Dict[str, Any]]:
    device_obj = resolve_device(device)
    checkpoint = torch.load(checkpoint_path, map_location=device_obj)
    model = build_model(checkpoint["config"]["model_name"], **checkpoint["model_kwargs"]).to(device_obj)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint["stoi"], checkpoint["itos"], checkpoint

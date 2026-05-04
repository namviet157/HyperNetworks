"""Train LSTM and HyperLSTM on the same corpus and save a comparison report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dynamic_hypernetwork.data import prepare_corpus
from dynamic_hypernetwork.training import TrainConfig, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare vanilla LSTM and HyperLSTM on the same dataset.")
    parser.add_argument("--text-file", type=str, default=None)
    parser.add_argument("--download-tinyshakespeare", action="store_true")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="artifacts/comparison")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--embedding-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--hyper-hidden-size", type=int, default=32)
    parser.add_argument("--hyper-embedding-size", type=int, default=8)
    parser.add_argument("--recurrent-dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--sample-every", type=int, default=100)
    parser.add_argument("--sample-length", type=int, default=240)
    parser.add_argument("--sample-temperature", type=float, default=0.9)
    parser.add_argument("--prompt", type=str, default="To be, or not to be")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def summary_table(results: list[dict]) -> str:
    lines = [
        "| Model | Params | Best Val Loss | Best Val PPL | Test Loss | Test PPL |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        lines.append(
            "| {model_name} | {num_parameters} | {best_val_loss:.4f} | {best_val_perplexity:.2f} | "
            "{test_loss:.4f} | {test_perplexity:.2f} |".format(**result)
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    corpus = prepare_corpus(
        text_file=args.text_file,
        download_tinyshakespeare=args.download_tinyshakespeare,
        data_dir=args.data_dir,
    )

    common_kwargs = dict(
        device=args.device,
        steps=args.steps,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        embedding_size=args.embedding_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        hyper_hidden_size=args.hyper_hidden_size,
        hyper_embedding_size=args.hyper_embedding_size,
        recurrent_dropout=args.recurrent_dropout,
        learning_rate=args.learning_rate,
        grad_clip=args.grad_clip,
        eval_every=args.eval_every,
        eval_batches=args.eval_batches,
        sample_every=args.sample_every,
        sample_length=args.sample_length,
        sample_temperature=args.sample_temperature,
        prompt=args.prompt,
        seed=args.seed,
    )

    results = []
    for model_name in ("lstm", "hyperlstm"):
        run_dir = output_dir / model_name
        config = TrainConfig(model_name=model_name, **common_kwargs)
        result = train_model(corpus=corpus, config=config, output_dir=run_dir)
        results.append(result)

    comparison = {
        "source_name": corpus.source_name,
        "results": results,
    }

    (output_dir / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    report = summary_table(results)
    (output_dir / "comparison.md").write_text(report + "\n", encoding="utf-8")
    print(report)
    print("[done] comparison saved to", output_dir / "comparison.md")


if __name__ == "__main__":
    main()

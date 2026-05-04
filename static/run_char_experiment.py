"""Train or sample a character-level LSTM / HyperLSTM demo."""

from __future__ import annotations

import argparse
from pathlib import Path

from dynamic_hypernetwork.data import prepare_corpus
from dynamic_hypernetwork.training import TrainConfig, load_model_from_checkpoint, resolve_device, train_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Submission-ready char-level demo for HyperNetworks.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train a model and save checkpoints.")
    train.add_argument("--model", choices=["lstm", "hyperlstm"], default="hyperlstm")
    train.add_argument("--text-file", type=str, default=None)
    train.add_argument("--download-tinyshakespeare", action="store_true")
    train.add_argument("--data-dir", type=str, default="data")
    train.add_argument("--output-dir", type=str, default="artifacts/run")
    train.add_argument("--device", type=str, default="cpu")
    train.add_argument("--steps", type=int, default=300)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--sequence-length", type=int, default=64)
    train.add_argument("--embedding-size", type=int, default=32)
    train.add_argument("--hidden-size", type=int, default=64)
    train.add_argument("--num-layers", type=int, default=1)
    train.add_argument("--dropout", type=float, default=0.0)
    train.add_argument("--hyper-hidden-size", type=int, default=32)
    train.add_argument("--hyper-embedding-size", type=int, default=8)
    train.add_argument("--recurrent-dropout", type=float, default=0.1)
    train.add_argument("--learning-rate", type=float, default=3e-3)
    train.add_argument("--grad-clip", type=float, default=1.0)
    train.add_argument("--eval-every", type=int, default=50)
    train.add_argument("--eval-batches", type=int, default=10)
    train.add_argument("--sample-every", type=int, default=100)
    train.add_argument("--sample-length", type=int, default=240)
    train.add_argument("--sample-temperature", type=float, default=0.9)
    train.add_argument("--prompt", type=str, default="To be, or not to be")
    train.add_argument("--seed", type=int, default=7)

    generate = subparsers.add_parser("generate", help="Generate text from a saved checkpoint.")
    generate.add_argument("--checkpoint", type=str, required=True)
    generate.add_argument("--device", type=str, default="cpu")
    generate.add_argument("--prompt", type=str, default="To be, or not to be")
    generate.add_argument("--length", type=int, default=400)
    generate.add_argument("--temperature", type=float, default=0.9)
    generate.add_argument("--output-file", type=str, default=None)

    return parser


def run_train(args: argparse.Namespace) -> None:
    corpus = prepare_corpus(
        text_file=args.text_file,
        download_tinyshakespeare=args.download_tinyshakespeare,
        data_dir=args.data_dir,
    )
    config = TrainConfig(
        model_name=args.model,
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
    summary = train_model(corpus=corpus, config=config, output_dir=args.output_dir)
    print("[done] summary saved to", Path(args.output_dir) / "summary.json")
    print(summary)


def run_generate(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    model, stoi, itos, checkpoint = load_model_from_checkpoint(args.checkpoint, device=str(device))
    text = model.sample(
        prompt=args.prompt,
        stoi=stoi,
        itos=itos,
        length=args.length,
        device=device,
        temperature=args.temperature,
    )
    print(f"[generate] source={checkpoint['source_name']} step={checkpoint['step']}")
    print(text)
    if args.output_file is not None:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "train":
        run_train(args)
    else:
        run_generate(args)


if __name__ == "__main__":
    main()

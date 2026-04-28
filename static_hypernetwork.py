#!/usr/bin/env python3
"""
CLI equivalent of static_hypernetwork.ipynb: full grid train, eval, benchmark.

Run from repository root:
  python static_hypernetwork.py train --full-grid
  python static_hypernetwork.py eval --full-grid --results-json runs/evaluation_results.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from solve.static_hypernet import Solver

DATASETS_ALL = ["mnist", "fashion_mnist", "cifar10", "svhn"]
MODELS_ALL = ["simplecnn", "wrn40_2", "resnet50"]

MODEL_TRAIN_DEFAULTS = {
    "simplecnn": {"max_epoch": 20, "learning_rate": 5e-4, "batch_size": 1024},
    "wrn40_2": {"max_epoch": 10, "learning_rate": 5e-4, "batch_size": 512},
    "resnet50": {"max_epoch": 10, "learning_rate": 5e-4, "batch_size": 256},
}

BENCHMARK_SETTINGS = [
    {
        "setting_name": "bench_lr5e4_default_bs",
        "overrides": {"max_epoch": 3, "learning_rate": 5e-4},
    },
    {
        "setting_name": "bench_lr1e3_bs256",
        "overrides": {"max_epoch": 3, "learning_rate": 1e-3, "batch_size": 256},
    },
]


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def component_name(hyper_mode: bool) -> str:
    return "static_hypernetwork" if hyper_mode else "baseline_conv"


def build_run_paths(
    dataset: str,
    model: str,
    hyper_mode: bool = False,
    setting_name: str | None = None,
) -> dict:
    run_name = f"{dataset}_{model}"
    if hyper_mode:
        run_name += "_hyper"
    if setting_name:
        run_name += f"_{setting_name}"
    run_root = Path("runs") / run_name
    return {
        "logpath": str(run_root / "logs"),
        "save_dir": str(run_root / "checkpoints"),
    }


def make_solver(
    dataset: str,
    model: str,
    hyper_mode: bool = False,
    setting_name: str | None = None,
    **kwargs,
) -> Solver:
    paths = build_run_paths(dataset, model, hyper_mode=hyper_mode, setting_name=setting_name)
    return Solver(
        dataset=dataset,
        model=model,
        hyper_mode=hyper_mode,
        logpath=paths["logpath"],
        save_dir=paths["save_dir"],
        **kwargs,
    )


def evaluate_best(
    dataset: str,
    model: str,
    hyper_mode: bool = False,
    setting_name: str | None = None,
    **kwargs,
) -> dict:
    solver = make_solver(
        dataset,
        model,
        hyper_mode=hyper_mode,
        setting_name=setting_name,
        **kwargs,
    )
    metrics = solver.evaluate_best_checkpoint()
    test_loss, test_acc = metrics["test"]
    return {
        "dataset": dataset,
        "model": model,
        "component": component_name(hyper_mode),
        "setting": setting_name or "main",
        "max_epoch": kwargs.get("max_epoch", ""),
        "learning_rate": kwargs.get("learning_rate", ""),
        "batch_size": kwargs.get("batch_size", ""),
        "test_loss": test_loss,
        "test_acc": test_acc,
    }


def print_results(results: list[dict]) -> None:
    header = (
        f"{'dataset':<15} {'model':<12} {'component':<22} {'setting':<28} "
        f"{'epochs':>6} {'lr':>10} {'batch':>7} {'test_loss':>10} {'test_acc':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in results:
        print(
            f"{row['dataset']:<15} {row['model']:<12} {row['component']:<22} "
            f"{row['setting']:<28} {str(row.get('max_epoch', '')):>6} "
            f"{str(row.get('learning_rate', '')):>10} {str(row.get('batch_size', '')):>7} "
            f"{row['test_loss']:>10.4f} {row['test_acc']:>10.4f}"
        )


def _parse_csv_list(value: str | None, allowed: list[str]) -> list[str]:
    if value is None or value.strip().lower() == "all":
        return list(allowed)
    parts = [p.strip() for p in value.split(",") if p.strip()]
    for p in parts:
        if p not in allowed:
            raise argparse.ArgumentTypeError(
                f"Invalid value {p!r}; allowed: {', '.join(allowed)}"
            )
    return parts


def _parse_hyper_modes(s: str) -> list[bool]:
    s = s.strip().lower()
    if s in ("both", "all"):
        return [False, True]
    if s in ("0", "baseline", "false", "no"):
        return [False]
    if s in ("1", "hyper", "true", "yes"):
        return [True]
    raise argparse.ArgumentTypeError(
        "hyper-modes must be one of: both, baseline, hyper (or 0/1)"
    )


def build_full_configs(
    datasets: list[str],
    models: list[str],
    hyper_modes: list[bool],
    setting_name: str = "main",
    overrides: dict | None = None,
) -> list[dict]:
    overrides = overrides or {}
    configs = []
    for dataset in datasets:
        for model in models:
            for hyper_mode in hyper_modes:
                config = {
                    "dataset": dataset,
                    "model": model,
                    "hyper_mode": hyper_mode,
                    "setting_name": setting_name,
                    **MODEL_TRAIN_DEFAULTS[model],
                    **overrides,
                }
                configs.append(config)
    return configs


def _solver_kwargs_from_config(config: dict) -> dict:
    return {
        "max_epoch": config["max_epoch"],
        "learning_rate": config["learning_rate"],
        "batch_size": config["batch_size"],
    }


def cmd_train(args: argparse.Namespace) -> None:
    datasets = _parse_csv_list(args.datasets, DATASETS_ALL)
    models = _parse_csv_list(args.models, MODELS_ALL)
    hyper_modes = _parse_hyper_modes(args.hyper_modes)
    setting_name = args.setting_name or "main"
    overrides = {}
    if args.max_epoch is not None:
        overrides["max_epoch"] = args.max_epoch
    if args.learning_rate is not None:
        overrides["learning_rate"] = args.learning_rate
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size

    configs = build_full_configs(
        datasets, models, hyper_modes, setting_name=setting_name, overrides=overrides
    )
    print(f"Training {len(configs)} run(s).")
    for config in configs:
        c = config.copy()
        sn = c.pop("setting_name")
        print("\n" + "=" * 100)
        print(
            f"Training dataset={c['dataset']} | model={c['model']} | "
            f"component={component_name(c['hyper_mode'])} | setting={sn}"
        )
        solver = make_solver(
            setting_name=None if sn == "main" else sn,
            show_sample=args.show_sample,
            show_filters=args.show_filters,
            **c,
        )
        solver.train()


def cmd_eval(args: argparse.Namespace) -> None:
    datasets = _parse_csv_list(args.datasets, DATASETS_ALL)
    models = _parse_csv_list(args.models, MODELS_ALL)
    hyper_modes = _parse_hyper_modes(args.hyper_modes)
    setting_name = args.setting_name or "main"
    configs = build_full_configs(datasets, models, hyper_modes, setting_name=setting_name)
    results = []
    for config in configs:
        c = config.copy()
        sn = c.pop("setting_name")
        kw = _solver_kwargs_from_config(c)
        result = evaluate_best(
            c["dataset"],
            c["model"],
            hyper_mode=c["hyper_mode"],
            setting_name=None if sn == "main" else sn,
            **kw,
        )
        results.append(result)
    print_results(results)
    if args.results_json:
        Path(args.results_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.results_json).write_text(
            json.dumps(results, indent=2, default=float), encoding="utf-8"
        )
        print(f"Wrote {args.results_json}")


def cmd_benchmark(args: argparse.Namespace) -> None:
    datasets = _parse_csv_list(args.datasets, DATASETS_ALL)
    models = _parse_csv_list(args.models, MODELS_ALL)
    hyper_modes = _parse_hyper_modes(args.hyper_modes)
    all_configs = []
    for setting in BENCHMARK_SETTINGS:
        all_configs.extend(
            build_full_configs(
                datasets,
                models,
                hyper_modes,
                setting_name=setting["setting_name"],
                overrides=setting["overrides"],
            )
        )
    print(f"Benchmarking {len(all_configs)} run(s).")
    results = []
    for config in all_configs:
        c = config.copy()
        sn = c.pop("setting_name")
        print("\n" + "=" * 100)
        print(
            f"Benchmark dataset={c['dataset']} | model={c['model']} | "
            f"component={component_name(c['hyper_mode'])} | setting={sn}"
        )
        solver = make_solver(
            setting_name=sn,
            show_sample=args.show_sample,
            show_filters=args.show_filters,
            **c,
        )
        solver.train()
        metrics = solver.evaluate_best_checkpoint()
        test_loss, test_acc = metrics["test"]
        results.append(
            {
                "dataset": c["dataset"],
                "model": c["model"],
                "component": component_name(c["hyper_mode"]),
                "setting": sn,
                "max_epoch": c["max_epoch"],
                "learning_rate": c["learning_rate"],
                "batch_size": c["batch_size"],
                "test_loss": test_loss,
                "test_acc": test_acc,
            }
        )
    print_results(results)
    if args.results_json:
        Path(args.results_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.results_json).write_text(
            json.dumps(results, indent=2, default=float), encoding="utf-8"
        )
        print(f"Wrote {args.results_json}")


def cmd_verify(args: argparse.Namespace) -> None:
    datasets = _parse_csv_list(args.datasets, DATASETS_ALL)
    models = _parse_csv_list(args.models, MODELS_ALL)
    hyper_modes = _parse_hyper_modes(args.hyper_modes)
    expected_main = len(datasets) * len(models) * len(hyper_modes)
    main = build_full_configs(datasets, models, hyper_modes, setting_name="main")
    bench_total = 0
    for setting in BENCHMARK_SETTINGS:
        bench_total += len(
            build_full_configs(
                datasets,
                models,
                hyper_modes,
                setting_name=setting["setting_name"],
                overrides=setting["overrides"],
            )
        )
    assert len(main) == expected_main
    assert bench_total == expected_main * len(BENCHMARK_SETTINGS)
    print(f"Main configs: {len(main)} (expected {expected_main})")
    print(f"Benchmark configs: {bench_total} (expected {expected_main * len(BENCHMARK_SETTINGS)})")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Static hypernetwork experiment grid (notebook parity).",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed.")

    sub = p.add_subparsers(dest="command", required=True)

    def add_grid_flags(sp: argparse.ArgumentParser) -> None:
        sp.add_argument(
            "--datasets",
            type=str,
            default="all",
            help="Comma-separated subset or 'all' (default).",
        )
        sp.add_argument(
            "--models",
            type=str,
            default="all",
            help="Comma-separated subset or 'all' (default).",
        )
        sp.add_argument(
            "--hyper-modes",
            type=str,
            default="both",
            help="both | baseline | hyper (aliases: 0 | 1).",
        )

    t = sub.add_parser("train", help="Train from scratch.")
    add_grid_flags(t)
    t.add_argument(
        "--setting-name",
        type=str,
        default="main",
        help="Run suffix under runs/; use 'main' for default paths (no extra suffix).",
    )
    t.add_argument("--max-epoch", type=int, default=None, help="Override epochs for all models.")
    t.add_argument("--learning-rate", type=float, default=None, help="Override LR for all models.")
    t.add_argument("--batch-size", type=int, default=None, help="Override batch size for all models.")
    t.add_argument("--show-sample", action="store_true", help="Show one training sample image.")
    t.add_argument("--show-filters", action="store_true", help="Show first conv filters after run.")
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("eval", help="Load best checkpoint and evaluate test split.")
    add_grid_flags(e)
    e.add_argument(
        "--setting-name",
        type=str,
        default="main",
        help="Must match the train run (default main).",
    )
    e.add_argument(
        "--results-json",
        type=str,
        default=None,
        help="Optional path to write evaluation results JSON.",
    )
    e.set_defaults(func=cmd_eval)

    b = sub.add_parser("benchmark", help="Full grid benchmark with preset hyperparameter settings.")
    add_grid_flags(b)
    b.add_argument(
        "--results-json",
        type=str,
        default=None,
        help="Optional path to write benchmark results JSON.",
    )
    b.add_argument("--show-sample", action="store_true")
    b.add_argument("--show-filters", action="store_true")
    b.set_defaults(func=cmd_benchmark)

    v = sub.add_parser("verify", help="Assert grid sizes without training.")
    add_grid_flags(v)
    v.set_defaults(func=cmd_verify)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    set_random_seed(args.seed)
    args.func(args)


if __name__ == "__main__":
    main()

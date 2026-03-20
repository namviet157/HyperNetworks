import argparse
from pathlib import Path
import random

import numpy as np
import tensorflow as tf


from solve.solver import Solver

DEFAULT_RUN_CONFIG = {
    "batch_size": 1024,
    "learning_rate": 5e-4,
    "lr_decay": 0.99,
    "val_split": 0.1,
    "save_best_only": False,
    "seed": 42,
}


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def build_run_paths(dataset, model, hyper_mode):
    run_name = f"{dataset}_{model}"
    if hyper_mode:
        run_name += "_hyper"
    run_root = Path("runs") / run_name
    return {
        "logpath": str(run_root / "logs"),
        "save_dir": str(run_root / "checkpoints"),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="StaticHyperNetwork runner")
    parser.add_argument("--dataset", default="mnist", choices=["mnist", "cifar10"])
    parser.add_argument("--model", default="simplecnn", choices=["simplecnn", "resnet50"])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--hyper-mode", action="store_true", default=False)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--eval-only", action="store_true", default=False)
    parser.add_argument("--show-sample", action="store_true", default=False)
    parser.add_argument("--show-filters", action="store_true", default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    run_paths = build_run_paths(args.dataset, args.model, args.hyper_mode)
    config = {**DEFAULT_RUN_CONFIG, **run_paths}

    set_random_seed(config["seed"])
    solver = Solver(
        dataset=args.dataset,
        model=args.model,
        max_epoch=args.epochs,
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        lr_decay=config["lr_decay"],
        hyper_mode=args.hyper_mode,
        logpath=config["logpath"],
        val_split=config["val_split"],
        save_dir=config["save_dir"],
        save_best_only=config["save_best_only"],
        resume=args.resume,
        eval_only=args.eval_only,
        seed=config["seed"],
        show_sample=args.show_sample,
        show_filters=args.show_filters,
    )
    solver.train()


if __name__ == "__main__":
    main()

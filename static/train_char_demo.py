"""Backward-compatible wrapper for the original quick demo command."""

from __future__ import annotations

import sys

from run_char_experiment import main


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv = [
            "train_char_demo.py",
            "train",
            "--model",
            "hyperlstm",
            "--output-dir",
            "artifacts/legacy_demo",
        ]
    elif sys.argv[1] not in {"train", "generate"}:
        sys.argv = ["train_char_demo.py", "train", "--model", "hyperlstm", *sys.argv[1:]]
    main()

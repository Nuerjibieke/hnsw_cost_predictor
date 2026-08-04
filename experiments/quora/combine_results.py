#!/usr/bin/env python3
"""Combine the Quora PGTuner and LightGBM rows into the paper table."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--ours", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Validate three Quora rows and write the reported columns."""
    args = parse_args()
    baseline = pd.read_csv(args.baseline).rename(columns={"splits": "folds"})
    ours = pd.read_csv(args.ours)
    table = pd.concat([baseline, ours], ignore_index=True)
    methods = ["PGTuner QPP", "Global context", "Ours"]
    if set(table.method.astype(str)) != set(methods) or not (table.folds == 7).all():
        raise RuntimeError("Quora result rows are incomplete")
    columns = [
        "method",
        "folds",
        "mae_log_mean",
        "mae_log_std",
        "qerror_median_mean",
        "qerror_median_std",
        "acc_1p2x_mean",
        "acc_1p2x_std",
        "acc_1p5x_mean",
        "acc_1p5x_std",
    ]
    table["method"] = pd.Categorical(table.method, methods, ordered=True)
    table = table.sort_values("method")[columns]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()

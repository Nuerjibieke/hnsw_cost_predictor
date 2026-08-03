#!/usr/bin/env python3
"""Summarize the fold-level ablation metrics reported in the paper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    """Build one compact mean-and-standard-deviation result table."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    rows = []
    metrics = {
        "mae_log": "mae_log",
        "median_qerror": "qerror_median",
        "p95_qerror": "qerror_p95",
        "acc_1p2_percent": "acc_1p2x",
        "acc_1p5_percent": "acc_1p5x",
    }
    for specification in plan["variants"]:
        path = args.training_root / specification["name"] / "fold_metrics.csv"
        folds = pd.read_csv(path)
        row = {
            "variant": specification["name"],
            "folds": len(folds),
        }
        for output_name, column in metrics.items():
            scale = 100.0 if output_name.endswith("_percent") else 1.0
            row[f"{output_name}_mean"] = float(folds[column].mean() * scale)
            row[f"{output_name}_std"] = float(folds[column].std(ddof=1) * scale)
        rows.append(row)
    table = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()

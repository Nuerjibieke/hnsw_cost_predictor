#!/usr/bin/env python3
"""Combine PGTuner and LightGBM metrics into the main paper result table."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def lightgbm_row(source: pd.DataFrame, variant: str, method: str) -> dict[str, object]:
    """Convert one ablation row to the main-table metric units."""
    matches = source[source.variant == variant]
    if len(matches) != 1:
        raise RuntimeError(f"missing ablation row: {variant}")
    row = matches.iloc[0]
    return {
        "method": method,
        "splits": int(row.folds),
        "mae_log_mean": row.mae_log_mean,
        "mae_log_std": row.mae_log_std,
        "qerror_median_mean": row.median_qerror_mean,
        "qerror_median_std": row.median_qerror_std,
        "qerror_p95_mean": row.p95_qerror_mean,
        "qerror_p95_std": row.p95_qerror_std,
        "acc_1p2x_mean": row.acc_1p2_percent_mean / 100.0,
        "acc_1p2x_std": row.acc_1p2_percent_std / 100.0,
        "acc_1p5x_mean": row.acc_1p5_percent_mean / 100.0,
        "acc_1p5x_std": row.acc_1p5_percent_std / 100.0,
    }


def main() -> None:
    """Validate and write the three methods reported in the main table."""
    args = parse_args()
    ablation = pd.read_csv(args.ablation)
    baseline = pd.read_csv(args.baseline)
    if len(baseline) != 1 or baseline.iloc[0].method != "PGTuner QPP":
        raise RuntimeError("PGTuner summary must contain one row")
    rows = [
        baseline.iloc[0].to_dict(),
        lightgbm_row(ablation, "context_only", "Global context only"),
        lightgbm_row(ablation, "final20", "Ours (final20)"),
    ]
    table = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()

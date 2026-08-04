#!/usr/bin/env python3
"""Compute the paper's within-setting Kendall rank correlation."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import kendalltau


def main() -> None:
    """Aggregate query-ranking correlations into the reported summary row."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    columns = [
        "fold_index",
        "dataset",
        "model_slug",
        "hnsw_ef_construction",
        "hnsw_M",
        "ef_search",
        "distance_computations",
        "pred_cost",
    ]
    frame = pd.read_parquet(args.predictions, columns=columns)
    group_columns = columns[:6]
    rows = []
    for key, group in frame.groupby(group_columns, observed=True, sort=True):
        tau = kendalltau(
            group["distance_computations"].to_numpy(float),
            group["pred_cost"].to_numpy(float),
            variant="b",
            nan_policy="omit",
        ).statistic
        rows.append(dict(zip(group_columns, key, strict=True), kendall_tau_b=tau))
    groups = pd.DataFrame(rows)
    splits = (
        groups.groupby(["fold_index", "dataset", "model_slug"], observed=True)
        .agg(kendall_tau_b_mean=("kendall_tau_b", "mean"))
        .reset_index()
    )
    summary = pd.DataFrame(
        [
            {
                "splits": len(splits),
                "mean_kendall_tau_b": splits["kendall_tau_b_mean"].mean(),
                "std_kendall_tau_b": splits["kendall_tau_b_mean"].std(ddof=1),
            }
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Combine the NSG and Vamana evaluations into the paper result table."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--nsg", type=Path, required=True)
    parser.add_argument("--vamana", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Validate four index rows and write them in paper order."""
    args = parse_args()
    table = pd.concat([pd.read_csv(args.nsg), pd.read_csv(args.vamana)], ignore_index=True)
    expected_order = [
        ("NSG", "Index context only"),
        ("NSG", "Index context with query-local features"),
        ("Vamana", "Index context only"),
        ("Vamana", "Index context with query-local features"),
    ]
    expected = set(expected_order)
    observed = set(map(tuple, table[["index", "feature_set"]].to_numpy()))
    if observed != expected or len(table) != 4:
        raise RuntimeError("NSG or Vamana result rows are incomplete")
    order = {key: index for index, key in enumerate(expected_order)}
    table["order"] = [order[(row["index"], row["feature_set"])] for _, row in table.iterrows()]
    table.sort_values("order").drop(columns="order").to_csv(args.output, index=False)


if __name__ == "__main__":
    main()

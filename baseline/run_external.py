#!/usr/bin/env python3
"""Evaluate PGTuner QPP on an external dataset without using its labels in training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hnsw_cost.model import metric
from model import train_predict
from run import CONFIG_KEYS, QUERY_COLUMNS, build_configuration_table, summarize


def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-table", type=Path, required=True)
    parser.add_argument("--training-features", type=Path, required=True)
    parser.add_argument("--external-table", type=Path, required=True)
    parser.add_argument("--external-features", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    return parser.parse_args()


def main() -> None:
    """Fit one leakage-free PGTuner fold for every external embedding model."""
    args = parse_args()
    plan = json.loads(args.config.read_text(encoding="utf-8"))
    train_queries = pd.read_parquet(args.training_table, columns=QUERY_COLUMNS)
    external_queries = pd.read_parquet(args.external_table, columns=QUERY_COLUMNS)
    if set(external_queries.dataset.astype(str)) != {"QuoraRetrieval"}:
        raise RuntimeError("the external baseline table must contain only QuoraRetrieval")
    if "QuoraRetrieval" in set(train_queries.dataset.astype(str)):
        raise RuntimeError("Quora labels leaked into baseline training")
    train_configs = build_configuration_table(train_queries, args.training_features)
    test_configs = build_configuration_table(external_queries, args.external_features)
    features = list(plan["features"])
    records = []
    for offset, held_model in enumerate(sorted(test_configs.model_slug.astype(str).unique())):
        train = train_configs[train_configs.model_slug.astype(str) != held_model]
        test = test_configs[test_configs.model_slug.astype(str) == held_model]
        predicted = np.maximum(
            train_predict(train, test, features, plan["training"], 78 + offset), 1.0
        )
        config_prediction = test[CONFIG_KEYS].copy()
        config_prediction["pred_log_cost"] = np.log1p(predicted)
        held_queries = external_queries[
            external_queries.model_slug.astype(str) == held_model
        ].merge(config_prediction, on=CONFIG_KEYS, how="left", validate="many_to_one")
        if held_queries.pred_log_cost.isna().any():
            raise RuntimeError(f"missing external baseline predictions for {held_model}")
        values = metric(held_queries.true_log_cost, held_queries.pred_log_cost)
        records.append(
            {
                "fold_index": 78 + offset,
                "held_dataset": "QuoraRetrieval",
                "held_model": held_model,
                **{key: values[key] for key in ["mae_log", "qerror_median", "qerror_p95", "acc_1p2x", "acc_1p5x"]},
            }
        )
    folds = pd.DataFrame(records)
    args.output_root.mkdir(parents=True, exist_ok=True)
    folds.to_csv(args.output_root / "fold_metrics.csv", index=False)
    summary = summarize(folds).rename(columns={"splits": "folds"})
    summary.to_csv(args.output_root / "summary.csv", index=False)


if __name__ == "__main__":
    main()

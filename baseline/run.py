#!/usr/bin/env python3
"""Train and evaluate the PGTuner QPP baseline under strict joint holdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from model import train_predict

from hnsw_cost.model import joint_holdout_masks, metric

CONFIG_KEYS = [
    "dataset",
    "model_slug",
    "hnsw_ef_construction",
    "hnsw_M",
    "ef_search",
]
QUERY_COLUMNS = [
    *CONFIG_KEYS,
    "query_offset",
    "distance_computations",
    "num_docs",
    "true_log_cost",
]


def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    repository = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training-table",
        type=Path,
        default=repository / "example/output/training_table.parquet",
    )
    parser.add_argument(
        "--pair-features",
        type=Path,
        default=repository / "example/output/baseline/pair_features.csv",
    )
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.json")
    )
    parser.add_argument(
        "--output-root", type=Path, default=repository / "example/output/baseline"
    )
    return parser.parse_args()


def fold_definitions(data: pd.DataFrame) -> list[tuple[int, str, str]]:
    """Enumerate the complete observed dataset-by-model holdout grid."""
    datasets = sorted(data.dataset.astype(str).unique())
    models = sorted(data.model_slug.astype(str).unique())
    observed = set(
        map(tuple, data[["dataset", "model_slug"]].drop_duplicates().to_numpy())
    )
    expected = {(dataset, model) for dataset in datasets for model in models}
    if observed != expected:
        raise RuntimeError("the baseline requires a complete dataset-by-model grid")
    return [
        (index, dataset, model)
        for index, (dataset, model) in enumerate(
            ((dataset, model) for dataset in datasets for model in models), start=1
        )
    ]


def build_configuration_table(
    queries: pd.DataFrame, pair_features: Path
) -> pd.DataFrame:
    """Aggregate query labels and attach dataset-level PGTuner features."""
    grouped = queries.groupby(CONFIG_KEYS, as_index=False, observed=True).agg(
        average_search_dc_counts=("distance_computations", "mean"),
        num_docs=("num_docs", "max"),
    )
    pair = pd.read_csv(pair_features)
    if pair.duplicated(["dataset", "model_slug"]).any():
        raise RuntimeError("the baseline feature table contains duplicate pairs")
    grouped = grouped.merge(
        pair, on=["dataset", "model_slug"], how="left", validate="many_to_one"
    )
    grouped["efConstruction"] = grouped.hnsw_ef_construction.astype(float)
    grouped["M"] = grouped.hnsw_M.astype(float)
    grouped["efSearch"] = grouped.ef_search.astype(float)
    if grouped["LID"].isna().any():
        raise RuntimeError("a dataset/model pair has no baseline feature row")
    if (grouped.num_docs != grouped.SIZE).any():
        raise RuntimeError("the baseline corpus size differs from the training table")
    counts_per_setting = queries.groupby(
        CONFIG_KEYS, observed=True
    ).query_offset.nunique()
    expected_counts = pair.set_index(["dataset", "model_slug"]).q_SIZE
    if any(
        int(value) != int(expected_counts.loc[index[:2]])
        for index, value in counts_per_setting.items()
    ):
        raise RuntimeError("an HNSW setting has incomplete baseline query labels")
    query_counts = queries.groupby(
        ["dataset", "model_slug"], observed=True
    ).query_offset.nunique()
    feature_counts = pair.set_index(["dataset", "model_slug"]).q_SIZE
    if any(
        int(feature_counts.loc[index]) != int(value)
        for index, value in query_counts.items()
    ):
        raise RuntimeError("the baseline query count differs from the training table")
    return grouped


def summarize(folds: pd.DataFrame) -> pd.DataFrame:
    """Build the macro fold summary reported in the paper."""
    row: dict[str, object] = {"method": "PGTuner QPP", "splits": len(folds)}
    for column in ["mae_log", "qerror_median", "qerror_p95", "acc_1p2x", "acc_1p5x"]:
        row[f"{column}_mean"] = float(folds[column].mean())
        row[f"{column}_std"] = float(folds[column].std(ddof=1))
    return pd.DataFrame([row])


def main() -> None:
    """Run every baseline fold and write compact result tables."""
    args = parse_args()
    plan = json.loads(args.config.read_text(encoding="utf-8"))
    queries = pd.read_parquet(args.training_table, columns=QUERY_COLUMNS)
    if queries.duplicated([*CONFIG_KEYS, "query_offset"]).any():
        raise RuntimeError("the training table contains duplicate query-setting rows")
    configurations = build_configuration_table(queries, args.pair_features)
    features = list(plan["features"])
    if not np.isfinite(configurations[features].to_numpy(float)).all():
        raise RuntimeError(
            "the baseline configuration table contains non-finite features"
        )

    records: list[dict[str, object]] = []
    for fold_index, held_dataset, held_model in fold_definitions(queries):
        train_mask, test_mask = joint_holdout_masks(
            configurations.dataset,
            configurations.model_slug,
            held_dataset,
            held_model,
        )
        train = configurations.loc[train_mask].copy()
        test = configurations.loc[test_mask].copy()
        if not len(train) or not len(test):
            raise RuntimeError(f"empty baseline fold {held_dataset}/{held_model}")
        predicted_cost = np.maximum(
            train_predict(train, test, features, plan["training"], fold_index), 1.0
        )
        if not np.isfinite(predicted_cost).all():
            raise RuntimeError(f"invalid baseline predictions in fold {fold_index}")
        config_prediction = test[CONFIG_KEYS].copy()
        config_prediction["pred_log_cost"] = np.log1p(predicted_cost)
        held_queries = queries[
            (queries.dataset == held_dataset) & (queries.model_slug == held_model)
        ].merge(config_prediction, on=CONFIG_KEYS, how="left", validate="many_to_one")
        if held_queries.pred_log_cost.isna().any():
            raise RuntimeError(
                "a configuration prediction could not be joined to its queries"
            )
        fold_metrics = metric(held_queries.true_log_cost, held_queries.pred_log_cost)
        records.append(
            {
                "fold_index": fold_index,
                "held_dataset": held_dataset,
                "held_model": held_model,
                **{
                    key: fold_metrics[key]
                    for key in [
                        "mae_log",
                        "qerror_median",
                        "qerror_p95",
                        "acc_1p2x",
                        "acc_1p5x",
                    ]
                },
            }
        )
        print(f"[ok] fold {fold_index}: {held_dataset} / {held_model}", flush=True)

    fold_frame = pd.DataFrame(records)
    summary = summarize(fold_frame)
    args.output_root.mkdir(parents=True, exist_ok=True)
    fold_frame.to_csv(args.output_root / "fold_metrics.csv", index=False)
    summary.to_csv(args.output_root / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

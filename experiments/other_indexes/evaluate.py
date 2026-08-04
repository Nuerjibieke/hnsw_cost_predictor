#!/usr/bin/env python3
"""Evaluate context-only and query-local predictors for NSG or Vamana labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from hnsw_cost.features import QUERY_FEATURES
from hnsw_cost.model import build_lgbm, metric


INDEX_CONFIG = {
    "NSG": {
        "context": ["nsg_R", "nsg_L", "nsg_C", "nsg_GK", "search_L", "num_docs"],
        "settings": ["nsg_R", "search_L"],
        "settings_per_pair": 8,
    },
    "Vamana": {
        "context": ["vamana_R", "vamana_Lbuild", "vamana_Lsearch", "num_docs"],
        "settings": ["vamana_R", "vamana_Lbuild", "vamana_Lsearch"],
        "settings_per_pair": 16,
    },
}
RANDOM_SEED = 42


def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", choices=sorted(INDEX_CONFIG), required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--training-table", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def read_labels(root: Path, setting_columns: list[str]) -> pd.DataFrame:
    """Load complete validated label shards for one index."""
    paths = sorted(root.glob("dataset=*/model=*.parquet"))
    if len(paths) != 77:
        raise RuntimeError(f"expected 77 label shards, found {len(paths)}")
    labels = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    keys = ["dataset", "model_slug", "query_offset", *setting_columns]
    if labels.duplicated(keys).any() or not (labels.counter_valid == 1).all():
        raise RuntimeError("duplicate or unvalidated other-index labels")
    if not np.isfinite(labels[["distance_computations", "true_log_cost"]]).all().all():
        raise RuntimeError("non-finite other-index labels")
    return labels


def read_query_features(training_table: Path) -> pd.DataFrame:
    """Read one copy of the 16 query-local features for every vector query."""
    columns = ["dataset", "model_slug", "query_offset", *QUERY_FEATURES]
    features = pd.read_parquet(training_table, columns=columns).drop_duplicates(
        ["dataset", "model_slug", "query_offset"]
    )
    if features.duplicated(["dataset", "model_slug", "query_offset"]).any():
        raise RuntimeError("duplicate query-local feature rows")
    return features


def training_weights(models: pd.Series) -> np.ndarray:
    """Give every embedding model equal total weight in a training fold."""
    counts = models.value_counts()
    weights = models.map(lambda name: 1.0 / float(counts[name])).to_numpy(float)
    return weights * (len(weights) / weights.sum())


def summarize(index_name: str, folds: pd.DataFrame) -> pd.DataFrame:
    """Build the two result rows reported for one graph index."""
    labels = {
        "context": "Index context only",
        "full": "Index context with query-local features",
    }
    rows = []
    for variant in ("context", "full"):
        group = folds[folds.variant == variant]
        rows.append(
            {
                "index": index_name,
                "feature_set": labels[variant],
                "mae_log_mean": float(group.mae_log.mean()),
                "mae_log_std": float(group.mae_log.std(ddof=1)),
                "p95_qerror_mean": float(group.qerror_p95.mean()),
                "p95_qerror_std": float(group.qerror_p95.std(ddof=1)),
                "acc_1p2_percent_mean": 100.0 * float(group.acc_1p2x.mean()),
                "acc_1p2_percent_std": 100.0 * float(group.acc_1p2x.std(ddof=1)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Run 77 strict joint holdouts and write the index result table."""
    args = parse_args()
    specification = INDEX_CONFIG[args.index]
    labels = read_labels(args.labels_root, specification["settings"])
    features = read_query_features(args.training_table)
    data = labels.merge(
        features,
        on=["dataset", "model_slug", "query_offset"],
        how="left",
        validate="many_to_one",
    )
    pairs = data[["dataset", "model_slug"]].drop_duplicates().sort_values(["dataset", "model_slug"])
    if len(pairs) != 77:
        raise RuntimeError(f"expected 77 other-index pairs, found {len(pairs)}")
    expected_rows = len(features) * int(specification["settings_per_pair"])
    if len(data) != expected_rows:
        raise RuntimeError(f"expected {expected_rows} label rows, found {len(data)}")
    variants = {
        "context": specification["context"],
        "full": [*specification["context"], *QUERY_FEATURES],
    }
    rows = []
    for fold_index, pair in enumerate(pairs.itertuples(index=False)):
        held_dataset, held_model = str(pair.dataset), str(pair.model_slug)
        train = data[(data.dataset != held_dataset) & (data.model_slug != held_model)]
        test = data[(data.dataset == held_dataset) & (data.model_slug == held_model)]
        if train.empty or test.empty:
            raise RuntimeError(f"empty other-index fold {fold_index}")
        weights = training_weights(train.model_slug)
        for variant, columns in variants.items():
            x_train = train[columns].to_numpy(dtype=float)
            x_test = test[columns].to_numpy(dtype=float)
            if not np.isfinite(x_train).all() or not np.isfinite(x_test).all():
                raise RuntimeError(f"non-finite other-index inputs in fold {fold_index}")
            predictor = build_lgbm(fold_index, len(columns), RANDOM_SEED)
            predictor.fit(
                x_train,
                train.true_log_cost.to_numpy(float),
                sample_weight=weights,
            )
            predicted = np.asarray(predictor.predict(x_test), dtype=float)
            rows.append(
                {
                    "fold_index": fold_index,
                    "held_dataset": held_dataset,
                    "held_model": held_model,
                    "variant": variant,
                    **metric(test.true_log_cost.to_numpy(float), predicted),
                }
            )
        print(f"[ok] {args.index} {held_dataset} / {held_model}", flush=True)
    folds = pd.DataFrame(rows)
    args.output_root.mkdir(parents=True, exist_ok=True)
    folds.to_csv(args.output_root / "fold_metrics.csv", index=False)
    summarize(args.index, folds).to_csv(args.output_root / "summary.csv", index=False)


if __name__ == "__main__":
    main()

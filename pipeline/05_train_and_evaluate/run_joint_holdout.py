#!/usr/bin/env python3
"""Train and evaluate every strict LODO+LOMO ablation fold."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OMP_THREAD_LIMIT", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from hnsw_cost.model import build_lgbm, joint_holdout_masks, metric, training_weights


IDENTITY_COLUMNS = [
    "dataset",
    "model_slug",
    "query_offset",
    "hnsw_ef_construction",
    "hnsw_M",
    "ef_search",
    "distance_computations",
    "num_docs",
    "true_log_cost",
]


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write a CSV file through an atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Write a Parquet file through an atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def fold_definitions(data: pd.DataFrame) -> list[tuple[int, str, str]]:
    """Enumerate the complete dataset-by-model holdout grid."""
    datasets = sorted(data.dataset.astype(str).unique())
    models = sorted(data.model_slug.astype(str).unique())
    observed = set(
        map(tuple, data[["dataset", "model_slug"]].drop_duplicates().to_numpy())
    )
    expected = {(dataset, model) for dataset in datasets for model in models}
    if observed != expected:
        raise RuntimeError(
            "training data must contain a complete dataset-by-model grid"
        )
    return [
        (index, dataset, model)
        for index, (dataset, model) in enumerate(
            ((dataset, model) for dataset in datasets for model in models),
            start=1,
        )
    ]


def read_training_table(path: Path, plan: dict) -> pd.DataFrame:
    """Read the identity and feature columns required by the plan."""
    features = list(
        dict.fromkeys(
            feature for variant in plan["variants"] for feature in variant["features"]
        )
    )
    return pd.read_parquet(
        path, columns=list(dict.fromkeys([*IDENTITY_COLUMNS, *features]))
    )


def run_worker(args: argparse.Namespace, plan: dict) -> None:
    """Run the holdout folds assigned to one worker."""
    data = read_training_table(args.training_table, plan)
    folds = [
        fold
        for fold in fold_definitions(data)
        if (fold[0] - 1) % args.num_workers == args.worker_id
    ]
    for fold_index, held_dataset, held_model in folds:
        train_mask, test_mask = joint_holdout_masks(
            data.dataset,
            data.model_slug,
            held_dataset,
            held_model,
        )
        train_index = np.flatnonzero(train_mask)
        test_index = np.flatnonzero(test_mask)
        if not len(train_index) or not len(test_index):
            raise RuntimeError(f"empty holdout fold {fold_index}")
        for specification in plan["variants"]:
            variant = str(specification["name"])
            token = (
                f"fold={fold_index:02d}__dataset={held_dataset.replace('/', '__')}"
                f"__model={held_model.replace('/', '__')}"
            )
            base = args.output_root / variant
            metric_path = base / "fold_metrics" / f"{token}.csv"
            prediction_path = base / "fold_predictions" / f"{token}.parquet"
            if metric_path.exists() and prediction_path.exists() and not args.force:
                continue

            features = specification["features"]
            train_features = data.iloc[train_index][features]
            medians = train_features.median(numeric_only=True).fillna(0.0)
            x_train = train_features.fillna(medians).astype(np.float32)
            x_test = data.iloc[test_index][features].fillna(medians).astype(np.float32)
            y_train = data.iloc[train_index].true_log_cost.to_numpy(float)
            y_test = data.iloc[test_index].true_log_cost.to_numpy(float)
            model = build_lgbm(fold_index, len(features), int(plan["random_seed_base"]))
            model.fit(
                x_train,
                y_train,
                sample_weight=training_weights(data.iloc[train_index].model_slug),
            )
            pred_log = np.asarray(model.predict(x_test), float)

            result = {
                "variant": variant,
                "fold_index": fold_index,
                "held_dataset": held_dataset,
                "held_model": held_model,
                **metric(y_test, pred_log),
            }
            predictions = data.iloc[test_index][IDENTITY_COLUMNS].copy()
            predictions["fold_index"] = fold_index
            predictions["pred_log_cost"] = pred_log
            predictions["pred_cost"] = np.maximum(np.expm1(pred_log), 1.0)
            atomic_csv(pd.DataFrame([result]), metric_path)
            atomic_parquet(predictions, prediction_path)
        print(f"[ok] fold {fold_index}: {held_dataset}/{held_model}", flush=True)


def merge_variant(
    root: Path, specification: dict, expected_folds: int, expected_rows: int
) -> None:
    """Merge the fold metrics and predictions for one feature variant."""
    base = root / specification["name"]
    metric_paths = sorted((base / "fold_metrics").glob("*.csv"))
    prediction_paths = sorted((base / "fold_predictions").glob("*.parquet"))
    if len(metric_paths) != expected_folds or len(prediction_paths) != expected_folds:
        raise RuntimeError(f"incomplete fold outputs for {specification['name']}")
    metrics = pd.concat([pd.read_csv(path) for path in metric_paths], ignore_index=True)
    predictions = pd.concat(
        [pd.read_parquet(path) for path in prediction_paths], ignore_index=True
    )
    if len(predictions) != expected_rows:
        raise RuntimeError(f"prediction row mismatch for {specification['name']}")
    atomic_csv(metrics.sort_values("fold_index"), base / "fold_metrics.csv")
    atomic_parquet(predictions, base / "predictions.parquet")


def run_merge(args: argparse.Namespace, plan: dict) -> None:
    """Validate and merge every completed feature variant."""
    data = pd.read_parquet(args.training_table, columns=["dataset", "model_slug"])
    expected_folds = len(fold_definitions(data))
    for specification in plan["variants"]:
        merge_variant(args.output_root, specification, expected_folds, len(data))
    print(f"Merged {len(plan['variants'])} variants across {expected_folds} folds.")


def main() -> None:
    """Run worker or merge mode from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["worker", "merge"])
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--training-table", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.num_workers <= 0 or not 0 <= args.worker_id < args.num_workers:
        raise SystemExit("invalid worker assignment")
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if args.mode == "worker":
        run_worker(args, plan)
    else:
        run_merge(args, plan)


if __name__ == "__main__":
    main()

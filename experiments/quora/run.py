#!/usr/bin/env python3
"""Train on the 11 paper datasets and evaluate the held-out Quora workload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hnsw_cost.model import build_lgbm, metric


IDENTITY = [
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


def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-table", type=Path, required=True)
    parser.add_argument("--quora-table", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/quora/ours.csv"))
    parser.add_argument(
        "--plan", type=Path, default=Path("configs/ablation_plan.json")
    )
    return parser.parse_args()


def summarize(folds: pd.DataFrame) -> pd.DataFrame:
    """Return the two macro rows reported for the LightGBM variants."""
    labels = {"context_only": "Global context", "final20": "Ours"}
    rows = []
    for variant in ("context_only", "final20"):
        selected = folds[folds.variant == variant]
        row: dict[str, object] = {"method": labels[variant], "folds": len(selected)}
        for column in ["mae_log", "qerror_median", "acc_1p2x", "acc_1p5x"]:
            row[f"{column}_mean"] = float(selected[column].mean())
            row[f"{column}_std"] = float(selected[column].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    """Fit seven leakage-free Quora folds and write their macro summary."""
    args = parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    variants = {
        row["name"]: row["features"]
        for row in plan["variants"]
        if row["name"] in {"context_only", "final20"}
    }
    if set(variants) != {"context_only", "final20"}:
        raise RuntimeError("required Quora variants are missing from the plan")
    all_features = list(dict.fromkeys(feature for values in variants.values() for feature in values))
    columns = list(dict.fromkeys([*IDENTITY, *all_features]))
    train_source = pd.read_parquet(args.training_table, columns=columns)
    quora = pd.read_parquet(args.quora_table, columns=columns)
    if "QuoraRetrieval" in set(train_source.dataset.astype(str)):
        raise RuntimeError("Quora must not appear in the training source")
    if set(quora.dataset.astype(str)) != {"QuoraRetrieval"}:
        raise RuntimeError("the external table must contain only QuoraRetrieval")
    models = sorted(quora.model_slug.astype(str).unique())
    if len(models) != 7:
        raise RuntimeError(f"expected seven Quora models, found {len(models)}")
    rows = []
    for offset, held_model in enumerate(models):
        train = train_source[train_source.model_slug.astype(str) != held_model]
        test = quora[quora.model_slug.astype(str) == held_model]
        if held_model in set(train.model_slug.astype(str)) or test.empty:
            raise RuntimeError(f"invalid Quora holdout for {held_model}")
        fold_index = 78 + offset
        for variant, features in variants.items():
            medians = train[features].median(numeric_only=True).fillna(0.0)
            x_train = train[features].fillna(medians).astype(np.float32)
            x_test = test[features].fillna(medians).astype(np.float32)
            if not np.isfinite(x_train).all().all() or not np.isfinite(x_test).all().all():
                raise RuntimeError(f"non-finite Quora inputs for {held_model}")
            model = build_lgbm(fold_index, len(features), int(plan["random_seed_base"]))
            model.fit(x_train, train.true_log_cost.to_numpy(float))
            prediction = np.asarray(model.predict(x_test), dtype=float)
            rows.append(
                {
                    "variant": variant,
                    "fold_index": fold_index,
                    "held_model": held_model,
                    **metric(test.true_log_cost.to_numpy(float), prediction),
                }
            )
        print(f"[ok] Quora / {held_model}", flush=True)
    folds = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    folds.to_csv(args.output.with_name("ours_folds.csv"), index=False)
    summarize(folds).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()

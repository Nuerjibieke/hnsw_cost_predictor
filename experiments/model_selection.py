#!/usr/bin/env python3
"""Evaluate embedding-model selection from predicted HNSW costs."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hnsw_cost.model import build_lgbm

SETTING_COLUMNS = ["hnsw_ef_construction", "hnsw_M", "ef_search"]
IDENTITY_COLUMNS = [
    "dataset",
    "model_slug",
    "query_offset",
    *SETTING_COLUMNS,
    "distance_computations",
    "true_log_cost",
]
BIN_LABELS = [
    "[1.00,1.02)",
    "[1.02,1.05)",
    "[1.05,1.10)",
    "[1.10,1.25)",
    "[1.25,+inf)",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-table", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=Path("configs/ablation_plan.json"))
    parser.add_argument("--variant", default="final20")
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/model_selection/summary.csv")
    )
    return parser.parse_args()


def load_features(plan_path: Path, variant: str) -> tuple[list[str], int]:
    """Load one named feature variant from the experiment plan."""
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    matches = [item["features"] for item in plan["variants"] if item["name"] == variant]
    if len(matches) != 1:
        raise RuntimeError(f"unknown feature variant: {variant}")
    return list(matches[0]), int(plan["random_seed_base"])


def load_table(path: Path, features: list[str]) -> pd.DataFrame:
    """Load and validate the rows needed for model selection."""
    columns = list(dict.fromkeys([*IDENTITY_COLUMNS, *features]))
    frame = pd.read_parquet(path, columns=columns)
    if frame.duplicated(
        ["dataset", "model_slug", "query_offset", *SETTING_COLUMNS]
    ).any():
        raise RuntimeError("training table contains duplicate query-setting rows")
    if not np.isfinite(
        frame[["distance_computations", "true_log_cost"]].to_numpy(float)
    ).all():
        raise RuntimeError("training labels contain non-finite values")
    if np.isinf(frame[features].to_numpy(float)).any():
        raise RuntimeError("training features contain infinite values")
    datasets = sorted(frame["dataset"].astype(str).unique())
    models = sorted(frame["model_slug"].astype(str).unique())
    pairs = set(
        map(tuple, frame[["dataset", "model_slug"]].drop_duplicates().to_numpy())
    )
    expected = {(dataset, model) for dataset in datasets for model in models}
    if pairs != expected:
        raise RuntimeError("training table must contain a complete dataset-model grid")
    if len(datasets) < 2 or len(models) < 3:
        raise RuntimeError(
            "model selection requires at least two datasets and three models"
        )
    expected_settings = set(
        map(tuple, frame[SETTING_COLUMNS].drop_duplicates().to_numpy())
    )
    for _, group in frame.groupby(["dataset", "model_slug"], observed=True):
        observed_settings = set(
            map(tuple, group[SETTING_COLUMNS].drop_duplicates().to_numpy())
        )
        if observed_settings != expected_settings:
            raise RuntimeError("dataset-model pairs do not share one HNSW setting grid")
    return frame


def ratio_bin(value: float) -> str:
    """Assign one measured cost ratio to the paper's reporting bin."""
    if value < 1.02:
        return BIN_LABELS[0]
    if value < 1.05:
        return BIN_LABELS[1]
    if value < 1.10:
        return BIN_LABELS[2]
    if value < 1.25:
        return BIN_LABELS[3]
    return BIN_LABELS[4]


def summarize(decisions: pd.DataFrame) -> pd.DataFrame:
    """Summarize selection accuracy and regret by measured cost ratio."""
    rows: list[dict[str, object]] = []
    for label in BIN_LABELS:
        group = decisions[decisions["cost_ratio"] == label]
        rows.append(
            {
                "cost_ratio": label,
                "decisions": len(group),
                "accuracy_percent": float(group["correct"].mean() * 100.0),
                "mean_regret": float(group["regret"].mean()),
            }
        )
    rows.append(
        {
            "cost_ratio": "Overall",
            "decisions": len(decisions),
            "accuracy_percent": float(decisions["correct"].mean() * 100.0),
            "mean_regret": float(decisions["regret"].mean()),
        }
    )
    return pd.DataFrame(rows)


def evaluate(data: pd.DataFrame, features: list[str], seed_base: int) -> pd.DataFrame:
    """Run the strict dataset-and-model-pair holdout protocol."""
    datasets = sorted(data["dataset"].astype(str).unique())
    models = sorted(data["model_slug"].astype(str).unique())
    decisions: list[dict[str, object]] = []
    episode = 0
    for dataset in datasets:
        for model_a, model_b in itertools.combinations(models, 2):
            episode += 1
            train = data[
                (data["dataset"] != dataset)
                & (~data["model_slug"].isin([model_a, model_b]))
            ]
            test = data[
                (data["dataset"] == dataset)
                & (data["model_slug"].isin([model_a, model_b]))
            ]
            if train.empty or test.empty:
                raise RuntimeError(
                    f"empty selection episode: {dataset}/{model_a}/{model_b}"
                )
            medians = train[features].median(numeric_only=True).fillna(0.0)
            x_train = train[features].fillna(medians).astype(np.float32)
            x_test = test[features].fillna(medians).astype(np.float32)
            model = build_lgbm(episode, len(features), seed_base)
            model.fit(x_train, train["true_log_cost"].to_numpy(float))
            predicted = np.maximum(np.expm1(model.predict(x_test)), 1.0)
            costs = test[
                ["model_slug", *SETTING_COLUMNS, "distance_computations"]
            ].copy()
            costs["predicted_cost"] = predicted
            means = (
                costs.groupby(["model_slug", *SETTING_COLUMNS], observed=True)
                .mean(numeric_only=True)
                .reset_index()
            )
            for _, group in means.groupby(SETTING_COLUMNS, observed=True):
                if set(group["model_slug"]) != {model_a, model_b}:
                    raise RuntimeError(
                        "a selection setting does not contain both models"
                    )
                row_a = group.loc[group["model_slug"] == model_a].iloc[0]
                row_b = group.loc[group["model_slug"] == model_b].iloc[0]
                true_a = float(row_a["distance_computations"])
                true_b = float(row_b["distance_computations"])
                pred_a = float(row_a["predicted_cost"])
                pred_b = float(row_b["predicted_cost"])
                selected = model_a if pred_a <= pred_b else model_b
                best_cost = min(true_a, true_b)
                if best_cost <= 0.0:
                    raise RuntimeError("measured selection costs must be positive")
                chosen_cost = true_a if selected == model_a else true_b
                decisions.append(
                    {
                        "cost_ratio": ratio_bin(max(true_a, true_b) / best_cost),
                        "correct": bool(np.isclose(chosen_cost, best_cost)),
                        "regret": chosen_cost / best_cost,
                    }
                )
    return pd.DataFrame(decisions)


def main() -> None:
    """Run model selection and write the paper result table."""
    args = parse_args()
    features, seed_base = load_features(args.plan, args.variant)
    decisions = evaluate(load_table(args.training_table, features), features, seed_base)
    if decisions.empty:
        raise RuntimeError("model selection produced no decisions")
    result = summarize(decisions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()

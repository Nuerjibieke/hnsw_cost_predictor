#!/usr/bin/env python3
"""Evaluate pairwise embedding-model selection with strict joint holdouts."""

from __future__ import annotations

import argparse
import itertools
import json
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

from hnsw_cost.fingerprint import fingerprint
from hnsw_cost.model import build_lgbm
from hnsw_cost.vector_io import sha256_file


SETTINGS = ["hnsw_ef_construction", "hnsw_M", "ef_search"]
IDENTITY = [
    "dataset",
    "model_slug",
    "query_offset",
    "distance_computations",
    "true_log_cost",
    *SETTINGS,
]
BINS = [1.0, 1.02, 1.05, 1.10, 1.25, np.inf]
LABELS = ["[1.00,1.02)", "[1.02,1.05)", "[1.05,1.10)", "[1.10,1.25)", "[1.25,+inf)"]


def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["worker", "merge"])
    parser.add_argument("--training-table", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=Path("configs/ablation_plan.json"))
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def episodes(data: pd.DataFrame) -> list[tuple[int, str, str, str]]:
    """Enumerate all dataset and unordered model-pair episodes."""
    datasets = sorted(data.dataset.astype(str).unique())
    models = sorted(data.model_slug.astype(str).unique())
    return [
        (index, dataset, model_a, model_b)
        for index, (dataset, (model_a, model_b)) in enumerate(
            itertools.product(datasets, itertools.combinations(models, 2)), start=1
        )
    ]


def episode_path(root: Path, episode: tuple[int, str, str, str]) -> Path:
    """Return the decision shard path for one episode."""
    index, dataset, model_a, model_b = episode
    dataset_name = dataset.replace("/", "__")
    model_a_name = model_a.replace("/", "__")
    model_b_name = model_b.replace("/", "__")
    filename = (
        f"episode={index:03d}__dataset={dataset_name}"
        f"__model_a={model_a_name}__model_b={model_b_name}.csv"
    )
    return root / "decisions" / filename


def lower_model(model_a: str, model_b: str, value_a: float, value_b: float) -> str:
    """Select the lower-cost model with a stable name-based tie break."""
    return min(
        (model_a, model_b),
        key=lambda model: (value_a if model == model_a else value_b, model),
    )


def evaluate_episode(
    data: pd.DataFrame,
    features: list[str],
    seed_base: int,
    episode: tuple[int, str, str, str],
    input_fingerprint: str,
) -> pd.DataFrame:
    """Fit one predictor and evaluate its 112 pairwise choices."""
    index, held_dataset, model_a, model_b = episode
    dataset_values = data.dataset.astype(str)
    model_values = data.model_slug.astype(str)
    train = data[
        (dataset_values != held_dataset)
        & (model_values != model_a)
        & (model_values != model_b)
    ]
    test = data[
        (dataset_values == held_dataset)
        & model_values.isin([model_a, model_b])
    ].copy()
    if train.empty or test.empty:
        raise RuntimeError(f"empty model-selection episode {index}")
    if held_dataset in set(train.dataset.astype(str)) or {
        model_a,
        model_b,
    } & set(train.model_slug.astype(str)):
        raise RuntimeError(f"holdout leakage in episode {index}")
    medians = train[features].median(numeric_only=True).fillna(0.0)
    x_train = train[features].fillna(medians).astype(np.float32)
    x_test = test[features].fillna(medians).astype(np.float32)
    model = build_lgbm(index, len(features), seed_base)
    model.fit(x_train, train.true_log_cost.to_numpy(float))
    test["pred_cost"] = np.maximum(np.expm1(model.predict(x_test)), 1.0)
    rows = []
    for values, group in test.groupby(SETTINGS, observed=True, sort=True):
        true = group.groupby("model_slug", observed=True).distance_computations.mean()
        predicted = group.groupby("model_slug", observed=True).pred_cost.mean()
        if set(true.index.astype(str)) != {model_a, model_b}:
            raise RuntimeError(f"candidate coverage mismatch in episode {index}")
        true_a, true_b = float(true[model_a]), float(true[model_b])
        predicted_a, predicted_b = float(predicted[model_a]), float(predicted[model_b])
        actual = lower_model(model_a, model_b, true_a, true_b)
        selected = lower_model(model_a, model_b, predicted_a, predicted_b)
        selected_true = true_a if selected == model_a else true_b
        rows.append(
            {
                "episode": index,
                "dataset": held_dataset,
                "model_a": model_a,
                "model_b": model_b,
                **dict(zip(SETTINGS, values)),
                "cost_ratio": max(true_a, true_b) / min(true_a, true_b),
                "correct": int(selected == actual),
                "regret": selected_true / min(true_a, true_b),
                "input_fingerprint": input_fingerprint,
            }
        )
    result = pd.DataFrame(rows)
    if len(result) != 112:
        raise RuntimeError(f"expected 112 decisions in episode {index}, found {len(result)}")
    return result


def summarize(decisions: pd.DataFrame) -> pd.DataFrame:
    """Group pairwise decisions into the cost-ratio rows reported in the paper."""
    decisions = decisions.copy()
    decisions["cost_ratio_group"] = pd.cut(
        decisions.cost_ratio, bins=BINS, labels=LABELS, right=False, include_lowest=True
    )
    rows = []
    for label in LABELS:
        group = decisions[decisions.cost_ratio_group == label]
        rows.append(
            {
                "cost_ratio": label,
                "decisions": len(group),
                "accuracy_percent": 100.0 * float(group.correct.mean()),
                "mean_regret": float(group.regret.mean()),
            }
        )
    rows.append(
        {
            "cost_ratio": "Overall",
            "decisions": len(decisions),
            "accuracy_percent": 100.0 * float(decisions.correct.mean()),
            "mean_regret": float(decisions.regret.mean()),
        }
    )
    return pd.DataFrame(rows)


def valid_shard(path: Path, expected_fingerprint: str) -> bool:
    """Return whether one cached episode matches the current inputs."""
    try:
        values = pd.read_csv(path, usecols=["input_fingerprint", *SETTINGS])
    except (OSError, ValueError):
        return False
    return (
        len(values) == 112
        and not values.duplicated(SETTINGS).any()
        and values[SETTINGS].nunique().tolist() == [4, 4, 7]
        and set(values.input_fingerprint.astype(str)) == {expected_fingerprint}
    )


def run_worker(
    args: argparse.Namespace,
    data: pd.DataFrame,
    features: list[str],
    seed_base: int,
    input_fingerprint: str,
) -> None:
    """Run the model-selection episodes assigned to one worker."""
    for episode in episodes(data):
        if (episode[0] - 1) % args.num_workers != args.worker_id:
            continue
        path = episode_path(args.output_root, episode)
        if path.exists() and not args.force and valid_shard(path, input_fingerprint):
            continue
        result = evaluate_episode(
            data, features, seed_base, episode, input_fingerprint
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(path, index=False)
        print(f"[ok] episode {episode[0]}: {episode[1]} / {episode[2]} / {episode[3]}", flush=True)


def run_merge(
    args: argparse.Namespace, data: pd.DataFrame, input_fingerprint: str
) -> None:
    """Merge complete decision shards and write the reported table."""
    paths = [episode_path(args.output_root, episode) for episode in episodes(data)]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise RuntimeError(f"missing {len(missing)} model-selection episode shards")
    invalid = [path for path in paths if not valid_shard(path, input_fingerprint)]
    if invalid:
        raise RuntimeError(f"stale or invalid model-selection shards: {len(invalid)}")
    decisions = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    if len(decisions) != 25_872 or not np.isfinite(decisions[["cost_ratio", "regret"]]).all().all():
        raise RuntimeError("invalid merged model-selection decisions")
    decisions = decisions.drop(columns=["input_fingerprint"])
    args.output_root.mkdir(parents=True, exist_ok=True)
    decisions.to_parquet(args.output_root / "decisions.parquet", index=False, compression="zstd")
    summarize(decisions).to_csv(args.output_root / "summary.csv", index=False)


def main() -> None:
    """Run or merge the complete model-selection evaluation."""
    args = parse_args()
    if args.num_workers <= 0 or not 0 <= args.worker_id < args.num_workers:
        raise SystemExit("invalid worker assignment")
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    features = next(row["features"] for row in plan["variants"] if row["name"] == "final20")
    input_fingerprint = fingerprint(
        {
            "training_table_sha256": sha256_file(args.training_table),
            "plan_sha256": sha256_file(args.plan),
            "script_sha256": sha256_file(Path(__file__)),
            "model_code_sha256": sha256_file(
                Path(__file__).resolve().parents[2] / "src/hnsw_cost/model.py"
            ),
            "lightgbm_version": metadata.version("lightgbm"),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "features": features,
            "seed_base": int(plan["random_seed_base"]),
        }
    )
    columns = list(dict.fromkeys([*IDENTITY, *features]))
    data = pd.read_parquet(args.training_table, columns=columns)
    if len(episodes(data)) != 231:
        raise RuntimeError("expected 231 model-selection episodes")
    if args.mode == "worker":
        run_worker(
            args,
            data,
            features,
            int(plan["random_seed_base"]),
            input_fingerprint,
        )
    else:
        run_merge(args, data, input_fingerprint)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Measure direct HNSW evaluation time and prediction time."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OMP_THREAD_LIMIT", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from hnsw_cost.features import compute_final_query_features, load_embeddings
from hnsw_cost.model import build_lgbm, joint_holdout_masks

SETTING_COLUMNS = ["hnsw_ef_construction", "hnsw_M", "ef_search"]
IDENTITY_COLUMNS = [
    "dataset",
    "model_slug",
    "query_offset",
    *SETTING_COLUMNS,
    "num_docs",
    "true_log_cost",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-table", type=Path, required=True)
    parser.add_argument("--vector-root", type=Path, required=True)
    parser.add_argument(
        "--lock", type=Path, default=Path("configs/vector_manifests.json")
    )
    parser.add_argument("--plan", type=Path, default=Path("configs/ablation_plan.json"))
    parser.add_argument("--grid", type=Path, default=Path("configs/hnsw_grid.json"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("artifacts/efficiency")
    )
    parser.add_argument("--datasets", nargs="*")
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--query-block-size", type=int, default=16)
    parser.add_argument("--prediction-warmups", type=int, default=3)
    parser.add_argument("--prediction-repeats", type=int, default=20)
    return parser.parse_args()


def load_final_features(plan_path: Path) -> tuple[list[str], int]:
    """Load the final predictor feature list and seed base."""
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    matches = [
        item["features"] for item in plan["variants"] if item["name"] == "final20"
    ]
    if len(matches) != 1:
        raise RuntimeError("the experiment plan must contain final20")
    return list(matches[0]), int(plan["random_seed_base"])


def load_table(path: Path, features: list[str]) -> pd.DataFrame:
    """Load and validate the full joint-holdout training table."""
    columns = list(dict.fromkeys([*IDENTITY_COLUMNS, *features]))
    frame = pd.read_parquet(path, columns=columns)
    if frame.duplicated(
        ["dataset", "model_slug", "query_offset", *SETTING_COLUMNS]
    ).any():
        raise RuntimeError("training table contains duplicate query-setting rows")
    if not np.isfinite(frame[["num_docs", "true_log_cost"]].to_numpy(float)).all():
        raise RuntimeError("training table contains non-finite labels")
    if np.isinf(frame[features].to_numpy(float)).any():
        raise RuntimeError("training table contains infinite features")
    datasets = sorted(frame["dataset"].astype(str).unique())
    models = sorted(frame["model_slug"].astype(str).unique())
    pairs = set(
        map(tuple, frame[["dataset", "model_slug"]].drop_duplicates().to_numpy())
    )
    expected = {(dataset, model) for dataset in datasets for model in models}
    if pairs != expected or len(datasets) < 2 or len(models) < 2:
        raise RuntimeError("training table must contain a complete joint-holdout grid")
    return frame


def fold_index(data: pd.DataFrame, dataset: str, model_slug: str) -> int:
    """Return the fixed joint-holdout fold number for one pair."""
    datasets = sorted(data["dataset"].astype(str).unique())
    models = sorted(data["model_slug"].astype(str).unique())
    pairs = [
        (item_dataset, item_model) for item_dataset in datasets for item_model in models
    ]
    try:
        return pairs.index((dataset, model_slug)) + 1
    except ValueError as error:
        raise RuntimeError(
            f"missing dataset-model pair: {dataset}/{model_slug}"
        ) from error


def locked_pairs(lock_path: Path) -> dict[tuple[str, str], dict]:
    """Load vector identities from the manifest lock."""
    pairs = json.loads(lock_path.read_text(encoding="utf-8"))["pairs"]
    return {(str(row["dataset"]), str(row["model_slug"])): row for row in pairs}


def direct_seconds(
    docs: np.ndarray, queries: np.ndarray, grid: dict[str, object]
) -> float:
    """Measure all index builds and batch searches for one vector pair."""
    import faiss

    if grid.get("metric") != "inner_product" or not grid.get("normalize_embeddings"):
        raise RuntimeError(
            "efficiency measurement requires normalized inner-product HNSW"
        )
    faiss.omp_set_num_threads(1)
    total = 0.0
    for ef_construction in grid["ef_construction_values"]:
        for m_value in grid["M_values"]:
            started = time.perf_counter()
            index = faiss.IndexHNSWFlat(
                docs.shape[1], int(m_value), faiss.METRIC_INNER_PRODUCT
            )
            index.hnsw.efConstruction = int(ef_construction)
            index.add(docs)
            total += time.perf_counter() - started
            for ef_search in grid["ef_search_values"]:
                index.hnsw.efSearch = int(ef_search)
                started = time.perf_counter()
                index.search(queries, int(grid["top_k"]))
                total += time.perf_counter() - started
    return total


def prediction_seconds(
    data: pd.DataFrame,
    dataset: str,
    model_slug: str,
    features: list[str],
    seed_base: int,
    queries: np.ndarray,
    docs: np.ndarray,
    settings: pd.DataFrame,
    query_block_size: int,
    warmups: int,
    repeats: int,
) -> float:
    """Train an excluded-time fold model and measure feature-to-prediction work."""
    train_mask, _ = joint_holdout_masks(
        data.dataset, data.model_slug, dataset, model_slug
    )
    train = data.loc[train_mask]
    if train.empty:
        raise RuntimeError(f"empty training fold: {dataset}/{model_slug}")
    medians = train[features].median(numeric_only=True).fillna(0.0)
    model = build_lgbm(fold_index(data, dataset, model_slug), len(features), seed_base)
    model.fit(
        train[features].fillna(medians).astype(np.float32),
        train["true_log_cost"].to_numpy(float),
    )
    started = time.perf_counter()
    query_features = compute_final_query_features(
        queries,
        docs,
        np.arange(len(queries), dtype=np.int64),
        query_block_size=query_block_size,
    )
    feature_time = time.perf_counter() - started
    started = time.perf_counter()
    inputs = settings.merge(query_features, how="cross")
    x_test = inputs[features].fillna(medians).astype(np.float32)
    assembly_time = time.perf_counter() - started
    for _ in range(warmups):
        model.predict(x_test, num_threads=1)
    times = []
    for _ in range(repeats):
        started = time.perf_counter()
        prediction = model.predict(x_test, num_threads=1)
        times.append(time.perf_counter() - started)
    if not np.isfinite(prediction).all():
        raise RuntimeError("prediction timing produced non-finite values")
    return feature_time + assembly_time + float(statistics.median(times))


def summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the compact paper-style runtime summary."""
    speedup = frame["speedup"].to_numpy(float)
    values = {
        "dataset_model_pairs": len(frame),
        "direct_seconds_min": frame["direct_seconds"].min(),
        "direct_seconds_max": frame["direct_seconds"].max(),
        "prediction_seconds_min": frame["prediction_seconds"].min(),
        "prediction_seconds_max": frame["prediction_seconds"].max(),
        "speedup_min": speedup.min(),
        "speedup_p10": np.quantile(speedup, 0.10),
        "speedup_median": np.median(speedup),
        "speedup_p90": np.quantile(speedup, 0.90),
        "speedup_max": speedup.max(),
    }
    return pd.DataFrame({"metric": list(values), "value": list(values.values())})


def main() -> None:
    """Measure every selected dataset-model pair and write two CSV files."""
    args = parse_args()
    if (
        args.query_block_size <= 0
        or args.prediction_warmups < 0
        or args.prediction_repeats <= 0
    ):
        raise SystemExit("invalid timing arguments")
    features, seed_base = load_final_features(args.plan)
    data = load_table(args.training_table, features)
    lock = locked_pairs(args.lock)
    grid = json.loads(args.grid.read_text(encoding="utf-8"))
    pairs = (
        data[["dataset", "model_slug"]]
        .drop_duplicates()
        .sort_values(["dataset", "model_slug"])
    )
    records: list[dict[str, object]] = []
    for pair in pairs.itertuples(index=False):
        dataset, model_slug = str(pair.dataset), str(pair.model_slug)
        if args.datasets and dataset not in args.datasets:
            continue
        if args.models and model_slug not in args.models:
            continue
        identity = lock.get((dataset, model_slug))
        if identity is None:
            raise RuntimeError(f"missing vector identity: {dataset}/{model_slug}")
        queries, docs, manifest = load_embeddings(
            args.vector_root,
            dataset,
            model_slug,
            revision=str(identity["revision"]),
            run_id=str(identity["run_id"]),
        )
        if (
            hashlib.sha256(manifest.read_bytes()).hexdigest()
            != identity["manifest_sha256"]
        ):
            raise RuntimeError(f"manifest hash mismatch: {dataset}/{model_slug}")
        if len(queries) != int(identity["num_queries"]) or len(docs) != int(
            identity["num_docs"]
        ):
            raise RuntimeError(f"vector cardinality mismatch: {dataset}/{model_slug}")
        observed_num_docs = data.loc[
            (data["dataset"] == dataset) & (data["model_slug"] == model_slug),
            "num_docs",
        ].unique()
        if len(observed_num_docs) != 1 or int(observed_num_docs[0]) != len(docs):
            raise RuntimeError(
                f"training-table corpus size mismatch: {dataset}/{model_slug}"
            )
        settings = pd.DataFrame(
            [
                (int(efc), int(m_value), int(ef_search), len(docs))
                for efc in grid["ef_construction_values"]
                for m_value in grid["M_values"]
                for ef_search in grid["ef_search_values"]
            ],
            columns=[*SETTING_COLUMNS, "num_docs"],
        )
        measured_settings = set(
            map(
                tuple,
                data.loc[
                    (data["dataset"] == dataset) & (data["model_slug"] == model_slug),
                    SETTING_COLUMNS,
                ]
                .drop_duplicates()
                .to_numpy(),
            )
        )
        expected_settings = set(map(tuple, settings[SETTING_COLUMNS].to_numpy()))
        if measured_settings != expected_settings:
            raise RuntimeError(f"HNSW setting grid mismatch: {dataset}/{model_slug}")
        direct = direct_seconds(docs, queries, grid)
        predicted = prediction_seconds(
            data,
            dataset,
            model_slug,
            features,
            seed_base,
            queries,
            docs,
            settings,
            args.query_block_size,
            args.prediction_warmups,
            args.prediction_repeats,
        )
        records.append(
            {
                "dataset": dataset,
                "model_slug": model_slug,
                "direct_seconds": direct,
                "prediction_seconds": predicted,
                "speedup": direct / predicted,
            }
        )
        print(f"[ok] {dataset}/{model_slug}", flush=True)
    result = pd.DataFrame(records)
    if result.empty:
        raise RuntimeError("no dataset-model pair matches the filters")
    args.output_root.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_root / "timing.csv", index=False)
    summary(result).to_csv(args.output_root / "summary.csv", index=False)


if __name__ == "__main__":
    main()

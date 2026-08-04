#!/usr/bin/env python3
"""Measure direct HNSW evaluation and ready-model prediction on the same grid."""

from __future__ import annotations

import argparse
import json
import os
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


def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-table", type=Path, required=True)
    parser.add_argument("--vector-root", type=Path, required=True)
    parser.add_argument("--lock", type=Path, default=Path("configs/vector_manifests.json"))
    parser.add_argument("--index-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=Path("configs/ablation_plan.json"))
    parser.add_argument("--grid", type=Path, default=Path("configs/hnsw_grid.json"))
    return parser.parse_args()


def fold_pairs(data: pd.DataFrame) -> list[tuple[int, str, str]]:
    """Enumerate the complete dataset and embedding-model holdout grid."""
    datasets = sorted(data.dataset.astype(str).unique())
    models = sorted(data.model_slug.astype(str).unique())
    observed = set(map(tuple, data[["dataset", "model_slug"]].drop_duplicates().to_numpy()))
    expected = {(dataset, model) for dataset in datasets for model in models}
    if observed != expected:
        raise RuntimeError("efficiency timing requires a complete pair grid")
    pairs = [(dataset, model) for dataset in datasets for model in models]
    return [(index, dataset, model) for index, (dataset, model) in enumerate(pairs, start=1)]


def pair_index_summaries(index_root: Path, dataset: str, model_slug: str) -> list[dict]:
    """Load the 16 HNSW build summaries for one vector pair."""
    summaries = []
    for path in sorted((index_root / "experiments").rglob("summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("task_name") == dataset
            and payload.get("model_slug") == model_slug
        ):
            summaries.append(payload)
    if len(summaries) != 16:
        raise RuntimeError(
            f"expected 16 indexes for {dataset}/{model_slug}, found {len(summaries)}"
        )
    return summaries


def time_direct_search(
    summaries: list[dict], queries: np.ndarray, ef_values: list[int]
) -> tuple[float, float]:
    """Time 16 saved index builds and 112 complete batch searches."""
    import faiss

    faiss.omp_set_num_threads(1)
    build_seconds = float(sum(float(row["build_seconds"]) for row in summaries))
    search_seconds = 0.0
    for row in summaries:
        index = faiss.read_index(str(row["index_path"]))
        for ef_search in ef_values:
            index.hnsw.efSearch = ef_search
            started = time.perf_counter()
            index.search(queries, int(row["k"]))
            search_seconds += time.perf_counter() - started
    return build_seconds, search_seconds


def fit_fold_model(
    data: pd.DataFrame,
    features: list[str],
    fold_index: int,
    dataset: str,
    model_slug: str,
    seed_base: int,
) -> tuple[object, pd.Series]:
    """Fit one joint-holdout predictor before the timed region."""
    train_mask, _ = joint_holdout_masks(
        data.dataset, data.model_slug, dataset, model_slug
    )
    train = data.loc[train_mask]
    medians = train[features].median(numeric_only=True).fillna(0.0)
    predictor = build_lgbm(fold_index, len(features), seed_base)
    predictor.fit(
        train[features].fillna(medians).astype(np.float32),
        train.true_log_cost.to_numpy(float),
    )
    return predictor, medians


def time_ready_prediction(
    predictor: object,
    medians: pd.Series,
    pair_rows: pd.DataFrame,
    queries: np.ndarray,
    docs: np.ndarray,
    features: list[str],
) -> tuple[float, float, float]:
    """Time query features, setting assembly, and ready-model inference."""
    started = time.perf_counter()
    computed = compute_final_query_features(
        queries, docs, np.arange(len(queries), dtype=np.int64), query_block_size=16
    )
    feature_seconds = time.perf_counter() - started
    started = time.perf_counter()
    context = pair_rows[["query_offset", "hnsw_ef_construction", "hnsw_M", "ef_search", "num_docs"]]
    assembled = context.merge(computed, on="query_offset", how="left", validate="many_to_one")
    inputs = assembled[features].fillna(medians).astype(np.float32)
    if not np.isfinite(inputs).all().all():
        raise RuntimeError("non-finite efficiency prediction inputs")
    assembly_seconds = time.perf_counter() - started
    started = time.perf_counter()
    predictor.predict(inputs, num_threads=1)
    inference_seconds = time.perf_counter() - started
    return feature_seconds, assembly_seconds, inference_seconds


def write_summary(timing: pd.DataFrame, output: Path) -> None:
    """Write the runtime ranges and speedup quantiles reported in the paper."""
    speedup = timing.speedup.to_numpy(float)
    rows = [
        ("dataset_model_pairs", len(timing)),
        ("direct_seconds_min", timing.direct_seconds.min()),
        ("direct_seconds_max", timing.direct_seconds.max()),
        ("prediction_seconds_min", timing.prediction_seconds.min()),
        ("prediction_seconds_max", timing.prediction_seconds.max()),
        ("speedup_min", np.min(speedup)),
        ("speedup_p10", np.quantile(speedup, 0.10)),
        ("speedup_median", np.median(speedup)),
        ("speedup_p90", np.quantile(speedup, 0.90)),
        ("speedup_max", np.max(speedup)),
    ]
    pd.DataFrame(rows, columns=["metric", "value"]).to_csv(output, index=False)


def main() -> None:
    """Run the single-threaded timing protocol for all 77 vector pairs."""
    args = parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    features = next(
        row["features"] for row in plan["variants"] if row["name"] == "final20"
    )
    columns = list(
        dict.fromkeys(
            ["dataset", "model_slug", "query_offset", "true_log_cost", *features]
        )
    )
    data = pd.read_parquet(args.training_table, columns=columns)
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    locked = {(row["dataset"], row["model_slug"]): row for row in lock["pairs"]}
    ef_values = json.loads(args.grid.read_text(encoding="utf-8"))["ef_search_values"]
    records = []
    for fold_index, dataset, model_slug in fold_pairs(data):
        pair = locked[(dataset, model_slug)]
        queries, docs, _ = load_embeddings(
            args.vector_root, dataset, model_slug, pair["revision"], pair["run_id"]
        )
        predictor, medians = fit_fold_model(
            data, features, fold_index, dataset, model_slug, int(plan["random_seed_base"])
        )
        pair_rows = data[(data.dataset == dataset) & (data.model_slug == model_slug)]
        feature_seconds, assembly_seconds, inference_seconds = time_ready_prediction(
            predictor,
            medians,
            pair_rows,
            queries,
            docs,
            features,
        )
        build_seconds, search_seconds = time_direct_search(
            pair_index_summaries(args.index_root, dataset, model_slug), queries, ef_values
        )
        direct = build_seconds + search_seconds
        prediction = feature_seconds + assembly_seconds + inference_seconds
        records.append(
            {
                "dataset": dataset,
                "model_slug": model_slug,
                "direct_seconds": direct,
                "prediction_seconds": prediction,
                "speedup": direct / prediction,
                "build_seconds": build_seconds,
                "search_seconds": search_seconds,
                "feature_seconds": feature_seconds,
                "assembly_seconds": assembly_seconds,
                "inference_seconds": inference_seconds,
            }
        )
        print(f"[ok] {dataset} / {model_slug}", flush=True)
    timing = pd.DataFrame(records)
    if len(timing) != 77:
        raise RuntimeError(f"expected 77 timing rows, found {len(timing)}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    timing.to_csv(args.output_root / "timing.csv", index=False)
    write_summary(timing, args.output_root / "summary.csv")


if __name__ == "__main__":
    main()

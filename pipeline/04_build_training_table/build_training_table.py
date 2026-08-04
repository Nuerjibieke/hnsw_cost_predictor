#!/usr/bin/env python3
"""Join measured per-query HNSW costs with the canonical query features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hnsw_cost.features import QUERY_FEATURES
from hnsw_cost.vector_io import sha256_file

SETTING_FEATURES = ["hnsw_ef_construction", "hnsw_M", "ef_search"]
KEYS = ["dataset", "model_slug", "query_offset"]
FULL_KEYS = [*KEYS, *SETTING_FEATURES]
LABEL_COLUMNS = [
    "dataset",
    "model_slug",
    "query_offset",
    "query_id",
    *SETTING_FEATURES,
    "distance_computations",
    "num_docs",
    "true_log_cost",
]


def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labels-root", type=Path, default=Path("artifacts/query_costs")
    )
    parser.add_argument(
        "--features-root", type=Path, default=Path("artifacts/query_features")
    )
    parser.add_argument(
        "--lock", type=Path, default=Path("configs/vector_manifests.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/training_table.parquet")
    )
    return parser.parse_args()


def pair_filename(dataset: str, model_slug: str) -> str:
    """Build the feature filename for a dataset/model pair."""
    return f"{dataset}__{model_slug.replace('/', '__')}.parquet"


def read_cost_labels(
    labels_root: Path, locked_pairs: dict[tuple[str, str], dict]
) -> pd.DataFrame:
    """Load locked per-query HNSW cost measurements."""
    paths = sorted((labels_root / "query_curves").rglob("query_curve.jsonl.gz"))
    if not paths:
        raise FileNotFoundError(f"no query_curve.jsonl.gz files under {labels_root}")
    parts: list[pd.DataFrame] = []
    for path in paths:
        success_path = path.parent / "_SUCCESS"
        if (
            not success_path.exists()
            or success_path.read_text(encoding="utf-8") != "ok\n"
        ):
            raise RuntimeError(f"incomplete cost measurement: {path}")
        meta = json.loads((path.parent / "meta.json").read_text(encoding="utf-8"))
        dataset = str(meta["task_name"])
        model_slug = str(meta["model_slug"])
        locked = locked_pairs.get((dataset, model_slug))
        if locked is None:
            continue
        if str(meta["manifest_sha256"]) != str(locked["manifest_sha256"]):
            raise RuntimeError(f"manifest identity mismatch for {dataset}/{model_slug}")
        if str(meta["curve_sha256"]) != sha256_file(path):
            raise RuntimeError(f"cost file hash mismatch: {path}")
        if str(meta["revision"]) != str(locked["revision"]) or str(
            meta["run_id"]
        ) != str(locked["run_id"]):
            raise RuntimeError(f"vector identity mismatch for {dataset}/{model_slug}")
        if int(meta["num_docs"]) != int(locked["num_docs"]) or int(
            meta["num_queries"]
        ) != int(locked["num_queries"]):
            raise RuntimeError(
                f"vector cardinality mismatch for {dataset}/{model_slug}"
            )
        frame = pd.read_json(path, lines=True, compression="gzip")
        if len(frame) != int(meta["curve_rows"]):
            raise RuntimeError(f"cost row count mismatch: {path}")
        frame = frame.rename(
            columns={
                "efConstruction": "hnsw_ef_construction",
                "M": "hnsw_M",
                "efSearch": "ef_search",
            }
        )
        required = [
            "query_offset",
            "query_id",
            *SETTING_FEATURES,
            "distance_computations",
        ]
        missing = [column for column in required if column not in frame]
        if missing:
            raise RuntimeError(f"missing label columns in {path}: {missing}")
        frame["query_id"] = frame["query_id"].astype(str)
        if "model_slug" in frame:
            measured_models = set(frame["model_slug"].astype(str).unique())
            if measured_models != {model_slug}:
                raise RuntimeError(
                    f"model identity mismatch in {path}: {sorted(measured_models)}"
                )
            frame = frame.drop(columns="model_slug")
        if "dataset" in frame:
            measured_datasets = set(frame["dataset"].astype(str).unique())
            if measured_datasets != {dataset}:
                raise RuntimeError(
                    f"dataset identity mismatch in {path}: {sorted(measured_datasets)}"
                )
            frame = frame.drop(columns="dataset")
        frame.insert(0, "model_slug", model_slug)
        frame.insert(0, "dataset", dataset)
        frame["num_docs"] = int(locked["num_docs"])
        frame["true_log_cost"] = np.log1p(frame.distance_computations.to_numpy(float))
        parts.append(frame[LABEL_COLUMNS])
    if not parts:
        raise RuntimeError(
            "none of the measured labels match the locked dataset/model pairs"
        )
    labels = pd.concat(parts, ignore_index=True)
    duplicate_count = int(labels.duplicated(FULL_KEYS).sum())
    if duplicate_count:
        raise RuntimeError(f"duplicate measured label keys: {duplicate_count}")
    numeric = labels[
        [*SETTING_FEATURES, "distance_computations", "num_docs", "true_log_cost"]
    ]
    if (
        not np.isfinite(numeric.to_numpy(float)).all()
        or (labels["distance_computations"] <= 0).any()
    ):
        raise RuntimeError("measured labels contain invalid numeric values")
    if labels.groupby(KEYS, observed=True).query_id.nunique().max() != 1:
        raise RuntimeError("query identifiers vary across HNSW settings")
    for (dataset, model_slug), locked in locked_pairs.items():
        offsets = set(
            labels.loc[
                (labels.dataset == dataset) & (labels.model_slug == model_slug),
                "query_offset",
            ].astype(int)
        )
        if offsets and offsets != set(range(int(locked["num_queries"]))):
            raise RuntimeError(f"query offsets mismatch for {dataset}/{model_slug}")
    setting_counts = (
        labels.groupby(KEYS, observed=True).size().astype(int).unique().tolist()
    )
    if len(setting_counts) != 1:
        raise RuntimeError("queries do not share one complete HNSW setting grid")
    return labels


def read_query_features(features_root: Path, pairs: list[dict]) -> pd.DataFrame:
    """Load and validate query-local feature tables."""
    parts: list[pd.DataFrame] = []
    for pair in pairs:
        dataset = str(pair["dataset"])
        model_slug = str(pair["model_slug"])
        path = features_root / pair_filename(dataset, model_slug)
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_parquet(path, columns=["query_offset", *QUERY_FEATURES])
        if len(frame) != int(pair["num_queries"]):
            raise RuntimeError(f"feature row count mismatch for {dataset}/{model_slug}")
        if set(frame.query_offset.astype(int)) != set(range(int(pair["num_queries"]))):
            raise RuntimeError(f"feature offsets mismatch for {dataset}/{model_slug}")
        if np.isinf(frame[QUERY_FEATURES].to_numpy(float)).any():
            raise RuntimeError(f"infinite query features for {dataset}/{model_slug}")
        frame.insert(0, "model_slug", model_slug)
        frame.insert(0, "dataset", dataset)
        parts.append(frame)
    features = pd.concat(parts, ignore_index=True)
    duplicate_count = int(features.duplicated(KEYS).sum())
    if duplicate_count:
        raise RuntimeError(f"duplicate query feature keys: {duplicate_count}")
    return features


def main() -> None:
    """Run the command-line workflow."""
    args = parse_args()
    pairs = json.loads(args.lock.read_text(encoding="utf-8"))["pairs"]
    locked_pairs = {(str(row["dataset"]), str(row["model_slug"])): row for row in pairs}
    labels = read_cost_labels(args.labels_root, locked_pairs)
    present_pairs = labels[["dataset", "model_slug"]].drop_duplicates()
    present_keys = set(map(tuple, present_pairs.to_numpy()))
    expected_keys = set(locked_pairs)
    if present_keys != expected_keys:
        missing = sorted(expected_keys - present_keys)
        extra = sorted(present_keys - expected_keys)
        raise RuntimeError(f"label pair mismatch; missing={missing}, extra={extra}")
    selected_pairs = [
        row
        for row in pairs
        if (str(row["dataset"]), str(row["model_slug"])) in present_keys
    ]
    features = read_query_features(args.features_root, selected_pairs)

    joined = labels.merge(
        features, on=KEYS, how="left", validate="many_to_one", indicator=True
    )
    missing_feature_rows = int((joined["_merge"] != "both").sum())
    if missing_feature_rows:
        raise RuntimeError(f"label rows without query features: {missing_feature_rows}")
    joined = joined.drop(columns="_merge")
    output_columns = [*LABEL_COLUMNS, *QUERY_FEATURES]
    joined = joined[output_columns].sort_values(FULL_KEYS).reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joined.to_parquet(args.output, index=False, compression="zstd")
    summary = {
        "rows": len(joined),
        "dataset_model_pairs": len(present_pairs),
        "unique_queries": int(features.shape[0]),
        "settings_per_pair": sorted(
            joined[["dataset", "model_slug", *SETTING_FEATURES]]
            .drop_duplicates()
            .groupby(["dataset", "model_slug"], observed=True)
            .size()
            .astype(int)
            .unique()
            .tolist()
        ),
        "model_features": [*SETTING_FEATURES, "num_docs", *QUERY_FEATURES],
        "model_feature_count": len(SETTING_FEATURES) + 1 + len(QUERY_FEATURES),
        "nonfinite_values_by_feature": {
            feature: int((~np.isfinite(joined[feature].to_numpy(float))).sum())
            for feature in QUERY_FEATURES
        },
        "output": str(args.output),
    }
    (args.output.parent / "training_table_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

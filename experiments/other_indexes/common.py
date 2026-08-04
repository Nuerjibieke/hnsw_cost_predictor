"""Share locked vector loading across the NSG and Vamana experiments."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from hnsw_cost.fingerprint import fingerprint
from hnsw_cost.features import load_embeddings


def locked_pairs(lock_path: Path) -> list[dict]:
    """Return the ordered dataset and model pairs from the vector lock."""
    pairs = json.loads(lock_path.read_text(encoding="utf-8"))["pairs"]
    keys = [(row["dataset"], row["model_slug"]) for row in pairs]
    if len(pairs) != 77 or len(set(keys)) != 77:
        raise RuntimeError("other-index experiments require the 77-pair vector lock")
    return pairs


def pair_embeddings(vector_root: Path, pair: dict) -> tuple[np.ndarray, np.ndarray]:
    """Load normalized query and document vectors for one locked pair."""
    queries, docs, _ = load_embeddings(
        vector_root,
        pair["dataset"],
        pair["model_slug"],
        pair["revision"],
        pair["run_id"],
    )
    expected = (int(pair["num_queries"]), int(pair["num_docs"]), int(pair["embedding_dim"]))
    observed = (len(queries), len(docs), docs.shape[1])
    if observed != expected or queries.shape[1] != docs.shape[1]:
        raise RuntimeError(f"locked vector shape mismatch: {observed} != {expected}")
    return queries, docs


def pair_output(root: Path, pair: dict) -> Path:
    """Return the label path for one dataset and model pair."""
    dataset = str(pair["dataset"]).replace("/", "__")
    model = str(pair["model_slug"]).replace("/", "__")
    return root / f"dataset={dataset}" / f"model={model}.parquet"


def pair_fingerprint(pair: dict, protocol: dict) -> str:
    """Bind one label shard to its vectors and counting protocol."""
    fields = {
        key: pair[key]
        for key in (
            "dataset",
            "model_slug",
            "revision",
            "run_id",
            "manifest_sha256",
            "num_docs",
            "num_queries",
            "embedding_dim",
        )
    }
    return fingerprint({"pair": fields, "protocol": protocol})


def valid_output(
    path: Path,
    expected_fingerprint: str,
    expected_rows: int,
    key_columns: list[str],
) -> bool:
    """Return whether a cached label shard matches its current inputs."""
    try:
        values = pd.read_parquet(
            path,
            columns=[
                "input_fingerprint",
                "distance_computations",
                "true_log_cost",
                *key_columns,
            ],
        )
    except (OSError, ValueError, KeyError):
        return False
    return (
        len(values) == expected_rows
        and not values.duplicated(key_columns).any()
        and (values.distance_computations > 0).all()
        and np.isfinite(values.true_log_cost.to_numpy(float)).all()
        and set(values.input_fingerprint.astype(str)) == {expected_fingerprint}
    )

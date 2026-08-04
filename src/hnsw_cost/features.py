"""Compute the paper's 16 query-local descriptors from embeddings."""

from __future__ import annotations

import math
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OMP_THREAD_LIMIT", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from .vector_io import l2_normalize, load_jsonl, resolve_path, verify_entry_files

EPS = 1e-8
MIN_K = 20
TOP_K = 40
PCTS = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05)
K_VALUES = (10, 20, 40)

DISTANCE_GROWTH_FEATURES = [
    "lid20",
    "adapt_lid_p005",
    "adapt_lid_p01",
    "adapt_lid_p02",
    "adapt_lid_p05",
    "adapt_lid_profile_mean",
    "adapt_lid_profile_max",
    "adapt_lid_profile_slope",
    "adapt_local_expansion_ratio_2r_p002",
    "adapt_local_expansion_ratio_2r_p01",
    "adapt_local_expansion_profile_mean",
]
DISTANCE_SEPARABILITY_FEATURES = ["rc10", "dsp10"]
ENTROPY_EFFECTIVE_RANK_FEATURES = ["qdir_erank_k10", "qdir_erank_k20", "qdir_erank_k40"]
QUERY_FEATURES = (
    DISTANCE_GROWTH_FEATURES
    + DISTANCE_SEPARABILITY_FEATURES
    + ENTROPY_EFFECTIVE_RANK_FEATURES
)


def choose_manifest(
    vector_root: Path,
    dataset: str,
    model_slug: str,
    revision: str | None = None,
    run_id: str | None = None,
) -> Path:
    """Select the complete embedding manifest matching a locked pair."""
    roots = [
        vector_root / "datasets" / dataset / "embeddings" / model_slug,
        vector_root / "embeddings" / model_slug,
        vector_root / dataset / "embeddings" / model_slug,
        vector_root.parent / "embeddings" / model_slug,
    ]
    complete: list[Path] = []
    for root in roots:
        for path in sorted(root.glob("*/*/manifest.jsonl")):
            if revision is not None and path.parent.parent.name != revision:
                continue
            if run_id is not None and path.parent.name != run_id:
                continue
            entries = [
                row
                for row in load_jsonl(path)
                if str(row.get("task_name", dataset)) == dataset
            ]
            kinds = {str(row.get("prompt_type")) for row in entries}
            if any("PromptType.query" in kind for kind in kinds) and any(
                "PromptType.document" in kind for kind in kinds
            ):
                complete.append(path)
    complete = sorted({path.resolve() for path in complete})
    if not complete:
        raise FileNotFoundError(f"no complete manifest for {dataset}/{model_slug}")
    if len(complete) != 1:
        raise RuntimeError(
            f"multiple manifests match {dataset}/{model_slug}: {len(complete)}"
        )
    return complete[0]


def split_manifest(path: Path, dataset: str) -> tuple[list[dict], list[dict]]:
    """Split manifest entries into ordered query and document batches."""
    queries: list[dict] = []
    docs: list[dict] = []
    for entry in load_jsonl(path):
        if str(entry.get("task_name", dataset)) != dataset:
            continue
        prompt_type = str(entry.get("prompt_type"))
        if "PromptType.query" in prompt_type:
            queries.append(entry)
        elif "PromptType.document" in prompt_type:
            docs.append(entry)
    queries.sort(key=lambda entry: int(entry.get("batch_index", 0)))
    docs.sort(key=lambda entry: int(entry.get("batch_index", 0)))
    return queries, docs


def load_embeddings(
    vector_root: Path,
    dataset: str,
    model_slug: str,
    revision: str | None = None,
    run_id: str | None = None,
) -> tuple[np.ndarray, np.ndarray, Path]:
    """Load and normalize the query and document embeddings for one pair."""
    manifest = choose_manifest(vector_root, dataset, model_slug, revision, run_id)
    query_entries, doc_entries = split_manifest(manifest, dataset)
    for entry in [*query_entries, *doc_entries]:
        verify_entry_files(entry, manifest)
    queries = l2_normalize(
        np.vstack(
            [
                np.load(resolve_path(entry["vectors_path"], manifest)).astype(
                    np.float32, copy=False
                )
                for entry in query_entries
            ]
        )
    )
    docs = l2_normalize(
        np.vstack(
            [
                np.load(resolve_path(entry["vectors_path"], manifest)).astype(
                    np.float32, copy=False
                )
                for entry in doc_entries
            ]
        )
    )
    return queries, docs, manifest


def pct_name(percent: float) -> str:
    """Convert a corpus fraction to its feature-name suffix."""
    return str(percent).replace("0.", "p").replace(".", "p")


def rank_for(num_docs: int, percent: float) -> int:
    """Convert a corpus fraction to a bounded neighbor rank."""
    return min(num_docs, max(MIN_K, math.ceil(num_docs * percent)))


def lid_mle(values: np.ndarray, radius: np.ndarray) -> np.ndarray:
    """Compute maximum-likelihood local intrinsic dimensionality."""
    values = np.maximum(values, EPS)
    radius = np.maximum(radius, EPS)
    mean_log = np.mean(np.log(values / radius[:, None]), axis=1)
    result = np.full(values.shape[0], np.nan, dtype=np.float64)
    valid = np.isfinite(mean_log) & (mean_log < -EPS)
    result[valid] = -1.0 / mean_log[valid]
    return result


def partition_positions(num_docs: int) -> list[int]:
    """Return the distance-order positions required by all features."""
    ranks = {10, 20, TOP_K, *(rank_for(num_docs, percent) for percent in PCTS)}
    return sorted({max(0, min(num_docs - 1, rank - 1)) for rank in ranks})


def topk_offsets(distances: np.ndarray, k: int = TOP_K) -> np.ndarray:
    """Select nearest vectors with document offsets breaking exact ties."""
    if distances.ndim != 2 or not 0 < k <= distances.shape[1]:
        raise ValueError("invalid nearest-neighbor selection shape or k")
    thresholds = np.partition(distances, kth=k - 1, axis=1)[:, k - 1]
    result = np.empty((distances.shape[0], k), dtype=np.int64)
    for row_index, (row, threshold) in enumerate(
        zip(distances, thresholds, strict=True)
    ):
        strictly_closer = np.flatnonzero(row < threshold)
        tied = np.flatnonzero(row == threshold)
        needed = k - len(strictly_closer)
        if needed < 0 or needed > len(tied):
            raise RuntimeError("nearest-neighbor boundary is not finite")
        selected = np.concatenate((strictly_closer, tied[:needed]))
        order = np.lexsort((selected, row[selected]))
        result[row_index] = selected[order]
    return result


def cosine_distances(queries: np.ndarray, docs: np.ndarray) -> np.ndarray:
    """Return cosine distances for already normalized embeddings."""
    scores = np.clip(queries @ docs.T, -1.0, 1.0).astype(np.float32, copy=False)
    return 1.0 - scores.astype(np.float64)


def compute_distance_features(
    distances: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Compute distance-growth and separability features for a query block."""
    num_docs = int(distances.shape[1])
    if num_docs < TOP_K:
        raise ValueError(f"at least {TOP_K} documents are required, got {num_docs}")
    ordered = np.partition(distances, kth=partition_positions(num_docs), axis=1).astype(
        np.float64, copy=False
    )
    top_offsets = topk_offsets(distances)
    top10 = ordered[:, :10]
    d10 = np.maximum(ordered[:, 9], EPS)
    mean_distance = distances.mean(axis=1)
    inverse_mass = (1.0 / np.maximum(distances, EPS)).sum(axis=1)

    output: dict[str, np.ndarray] = {
        "rc10": mean_distance / d10,
        "dsp10": top10.mean(axis=1)
        * (inverse_mass - (1.0 / np.maximum(top10, EPS)).sum(axis=1))
        / num_docs,
        "lid20": lid_mle(ordered[:, :19], ordered[:, 19]),
    }
    lids: list[np.ndarray] = []
    expansions: list[np.ndarray] = []
    for percent in PCTS:
        name = pct_name(percent)
        rank = rank_for(num_docs, percent)
        radius = np.maximum(ordered[:, rank - 1], EPS)
        lid = lid_mle(ordered[:, : rank - 1], radius)
        expansion = (distances <= (2.0 * radius)[:, None]).sum(axis=1) / max(rank, 1)
        output[f"adapt_lid_{name}"] = lid
        output[f"adapt_local_expansion_ratio_2r_{name}"] = expansion
        lids.append(lid)
        expansions.append(expansion)

    lid_profile = np.vstack(lids).T.astype(np.float64)
    lid_mean = np.nanmean(lid_profile, axis=1)
    positions = np.arange(lid_profile.shape[1], dtype=np.float64)
    centered = positions - positions.mean()
    output["adapt_lid_profile_mean"] = lid_mean
    output["adapt_lid_profile_max"] = np.nanmax(lid_profile, axis=1)
    output["adapt_lid_profile_slope"] = np.nansum(
        (lid_profile - lid_mean[:, None]) * centered[None, :], axis=1
    ) / float((centered**2).sum())
    output["adapt_local_expansion_profile_mean"] = np.nanmean(
        np.vstack(expansions).T.astype(np.float64),
        axis=1,
    )
    names = DISTANCE_GROWTH_FEATURES + DISTANCE_SEPARABILITY_FEATURES
    return {name: output[name] for name in names}, top_offsets


def effective_rank_from_eigenvalues(values: np.ndarray) -> float:
    """Compute entropy effective rank from eigenvalues."""
    values = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    total = float(values.sum())
    if total <= 1e-12:
        return math.nan
    probabilities = np.sort(values)[::-1] / total
    return float(
        math.exp(-float(np.sum(probabilities * np.log(probabilities + 1e-30))))
    )


def spectrum_effective_rank(vectors: np.ndarray) -> float:
    """Compute effective rank from a row-vector Gram matrix."""
    return effective_rank_from_eigenvalues(np.linalg.eigvalsh(vectors @ vectors.T))


def normalized_direction_effective_rank(
    query: np.ndarray, docs: np.ndarray, k: int
) -> float:
    """Compute effective rank of normalized query-centered directions."""
    centered = (
        docs[:k].astype(np.float64, copy=False)
        - query.astype(np.float64, copy=False)[None, :]
    )
    norms = np.linalg.norm(centered, axis=1)
    return spectrum_effective_rank(centered / np.maximum(norms[:, None], 1e-12))


def compute_final_query_features(
    queries: np.ndarray,
    docs: np.ndarray,
    query_offsets: np.ndarray,
    query_block_size: int = 16,
) -> pd.DataFrame:
    """Compute all 16 paper features for selected queries."""
    query_offsets = np.asarray(query_offsets, dtype=np.int64)
    if queries.ndim != 2 or docs.ndim != 2 or queries.shape[1] != docs.shape[1]:
        raise ValueError("query and document vectors must have one common dimension")
    if query_block_size <= 0 or not len(query_offsets):
        raise ValueError("query offsets and block size must be positive")
    if (
        np.any(query_offsets < 0)
        or np.any(query_offsets >= len(queries))
        or len(np.unique(query_offsets)) != len(query_offsets)
    ):
        raise ValueError("query offsets must be unique and in range")
    top_offset_parts: list[np.ndarray] = []
    feature_parts: list[pd.DataFrame] = []
    for start in range(0, len(query_offsets), query_block_size):
        end = min(start + query_block_size, len(query_offsets))
        distances = cosine_distances(queries[query_offsets[start:end]], docs)
        values, top_offsets = compute_distance_features(distances)
        part = pd.DataFrame(values)
        part.insert(0, "query_offset", query_offsets[start:end])
        feature_parts.append(part)
        top_offset_parts.append(top_offsets)
    features = pd.concat(feature_parts, ignore_index=True)
    top_offsets = np.vstack(top_offset_parts)

    rank_rows: list[dict[str, float]] = []
    for row_index, query_offset in enumerate(query_offsets):
        near_docs = docs[top_offsets[row_index]]
        record: dict[str, float] = {"query_offset": int(query_offset)}
        for k in K_VALUES:
            record[f"qdir_erank_k{k}"] = normalized_direction_effective_rank(
                queries[query_offset],
                near_docs,
                k,
            )
        rank_rows.append(record)
    features = features.merge(
        pd.DataFrame(rank_rows), on="query_offset", validate="one_to_one"
    )
    return features[["query_offset", *QUERY_FEATURES]]

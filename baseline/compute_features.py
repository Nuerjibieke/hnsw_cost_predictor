#!/usr/bin/env python3
"""Compute the dataset-level features used by the PGTuner QPP baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from hnsw_cost.vector_io import load_jsonl, load_vectors_and_rows, resolve_path


FEATURE_COLUMNS = [
    "dataset",
    "model_slug",
    "SIZE",
    "q_SIZE",
    "DIM",
    "LID",
    "Sum_K_MinDist",
    "Sum_K_MaxDist",
    "Sum_K_StdDist",
    "q_K_MinRatio",
    "q_K_MeanRatio",
    "q_K_MaxRatio",
    "q_K_StdRatio",
]


def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    repository = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vector-root", type=Path, default=repository / "example/data/vectors"
    )
    parser.add_argument(
        "--lock", type=Path, default=repository / "example/vector_manifests.json"
    )
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.json")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repository / "example/output/baseline/pair_features.csv",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tasks", nargs="*")
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def select_device(name: str) -> str:
    """Select CUDA when available or fall back to the CPU."""
    import torch

    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return name


def vector_directories(pair: dict, vector_root: Path) -> tuple[Path, Path]:
    """Resolve the document and query directories for one locked pair."""
    manifest = vector_root / str(pair["manifest_relative_path"])
    if not manifest.exists():
        raise FileNotFoundError(manifest)
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if digest != str(pair["manifest_sha256"]):
        raise RuntimeError(
            f"manifest hash mismatch for {pair['dataset']}/{pair['model_slug']}"
        )
    entries = [
        entry
        for entry in load_jsonl(manifest)
        if str(entry.get("task_name", pair["dataset"])) == str(pair["dataset"])
    ]
    document_dirs = {
        resolve_path(entry["vectors_path"], manifest).parent
        for entry in entries
        if "PromptType.document" in str(entry.get("prompt_type"))
    }
    query_dirs = {
        resolve_path(entry["vectors_path"], manifest).parent
        for entry in entries
        if "PromptType.query" in str(entry.get("prompt_type"))
    }
    if len(document_dirs) != 1 or len(query_dirs) != 1:
        raise RuntimeError(
            f"incomplete manifest for {pair['dataset']}/{pair['model_slug']}"
        )
    return document_dirs.pop(), query_dirs.pop()


def exact_topk_l2(
    base_vectors: np.ndarray,
    query_vectors: np.ndarray,
    k: int,
    batch_size: int,
    device: str,
    exclude_aligned_self: bool = False,
) -> np.ndarray:
    """Return exact batched Euclidean distances to the nearest vectors."""
    import torch

    if k > len(base_vectors):
        raise ValueError(f"k={k} exceeds the base size {len(base_vectors)}")
    base = torch.as_tensor(base_vectors, dtype=torch.float32, device=device)
    base_norm = (base * base).sum(dim=1).reshape(1, -1)
    parts: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(query_vectors), batch_size):
            end = min(start + batch_size, len(query_vectors))
            query = torch.as_tensor(
                query_vectors[start:end], dtype=torch.float32, device=device
            )
            query_norm = (query * query).sum(dim=1).reshape(-1, 1)
            squared = query_norm + base_norm - 2.0 * (query @ base.T)
            squared.clamp_(min=0.0)
            if exclude_aligned_self:
                rows = torch.arange(end - start, device=device)
                columns = torch.arange(start, end, device=device)
                squared[rows, columns] = float("inf")
            nearest = torch.topk(squared, k=k, largest=False, dim=1).values
            parts.append(torch.sqrt(nearest).cpu().numpy())
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return np.concatenate(parts, axis=0)


def dataset_structure(
    unique_documents: np.ndarray,
    neighbors: int,
    batch_size: int,
    device: str,
) -> dict[str, float]:
    """Compute the three PGTuner dataset-structure statistics."""
    distances = exact_topk_l2(
        unique_documents, unique_documents, neighbors + 1, batch_size, device
    )
    mean_distance = float(distances[:, 1:-1].mean())
    sums = (distances / max(mean_distance, 1e-12))[:, 1:-1].sum(axis=1)
    return {
        "Sum_K_MinDist": float(np.min(sums)),
        "Sum_K_MaxDist": float(np.max(sums)),
        "Sum_K_StdDist": float(np.std(sums)),
    }


def query_difficulty(
    documents: np.ndarray,
    queries: np.ndarray,
    neighbors: int,
    query_neighbors: int,
    batch_size: int,
    device: str,
) -> dict[str, float]:
    """Compute the four PGTuner query-difficulty ratio statistics."""
    count = min(query_neighbors, len(documents))
    if count <= neighbors:
        raise ValueError("the corpus must contain more than the near-neighbor count")
    distances = exact_topk_l2(documents, queries, count, batch_size, device)
    near = distances[:, :neighbors].mean(axis=1)
    background = distances[:, neighbors:].mean(axis=1)
    valid = ~np.isclose(background, 0.0)
    ratios = near[valid] / background[valid]
    if not len(ratios):
        raise RuntimeError("all query-difficulty denominators are zero")
    return {
        "q_K_MinRatio": float(np.min(ratios)),
        "q_K_MeanRatio": float(np.mean(ratios)),
        "q_K_MaxRatio": float(np.max(ratios)),
        "q_K_StdRatio": float(np.std(ratios)),
    }


def global_lid(unique_documents: np.ndarray, batch_size: int, device: str) -> float:
    """Estimate and round the PGTuner global intrinsic dimension."""
    standardized = unique_documents.astype(np.float64, copy=True)
    standardized -= np.mean(standardized, axis=0)
    standardized /= np.std(standardized, axis=0) + 1e-7
    standardized = standardized.astype(np.float32, copy=False)
    distances = exact_topk_l2(
        standardized, standardized, 13, batch_size, device, exclude_aligned_self=True
    )
    log_distances = np.log(distances[:, :12] + 1e-5)
    indices = np.arange(6, 13)
    cumulative = np.cumsum(log_distances, axis=1)
    estimate = np.mean(
        -(indices - 2)
        / (
            cumulative[:, 5:12]
            - log_distances[:, 5:12] * np.tile(indices, (len(standardized), 1))
            + 1e-5
        )
    )
    return float(math.floor(float(estimate) + 0.5))


def compute_pair(
    pair: dict, vector_root: Path, specification: dict, device: str
) -> dict:
    """Compute every PGTuner feature for one dataset/model pair."""
    document_dir, query_dir = vector_directories(pair, vector_root)
    documents, _ = load_vectors_and_rows(document_dir)
    queries, _ = load_vectors_and_rows(query_dir)
    expected_shape = (int(pair["num_docs"]), int(pair["embedding_dim"]))
    if documents.shape != expected_shape:
        raise RuntimeError(
            f"document shape mismatch for {pair['dataset']}/{pair['model_slug']}"
        )
    if queries.shape != (int(pair["num_queries"]), int(pair["embedding_dim"])):
        raise RuntimeError(
            f"query shape mismatch for {pair['dataset']}/{pair['model_slug']}"
        )
    unique_documents = np.unique(documents, axis=0).astype(np.float32, copy=False)
    neighbors = int(specification["neighbors"])
    if len(unique_documents) <= neighbors or len(unique_documents) < 14:
        raise RuntimeError(
            "the corpus is too small for the PGTuner feature definitions"
        )
    result = {
        "dataset": str(pair["dataset"]),
        "model_slug": str(pair["model_slug"]),
        "SIZE": len(documents),
        "q_SIZE": len(queries),
        "DIM": int(documents.shape[1]),
        "LID": global_lid(
            unique_documents, int(specification["lid_batch_size"]), device
        ),
        **dataset_structure(
            unique_documents, neighbors, int(specification["ds_batch_size"]), device
        ),
        **query_difficulty(
            documents,
            queries,
            neighbors,
            int(specification["query_neighbors"]),
            int(specification["dr_batch_size"]),
            device,
        ),
    }
    if not np.isfinite([float(result[column]) for column in FEATURE_COLUMNS[2:]]).all():
        raise RuntimeError("a baseline feature is non-finite")
    return result


def main() -> None:
    """Compute and save the selected locked-pair feature rows."""
    args = parse_args()
    plan = json.loads(args.config.read_text(encoding="utf-8"))["feature_extraction"]
    pairs = json.loads(args.lock.read_text(encoding="utf-8"))["pairs"]
    pairs = [
        pair
        for pair in pairs
        if (not args.tasks or str(pair["dataset"]) in args.tasks)
        and (
            not args.models
            or any(
                value.lower() in str(pair["model_slug"]).lower()
                for value in args.models
            )
        )
    ]
    if not pairs:
        raise RuntimeError("no locked pairs match the requested filters")
    device = select_device(args.device)
    if args.output.exists() and not args.force:
        existing = pd.read_csv(args.output)
        if (
            list(existing.columns) != FEATURE_COLUMNS
            or existing.duplicated(["dataset", "model_slug"]).any()
        ):
            raise RuntimeError(f"invalid existing feature table: {args.output}")
    else:
        existing = pd.DataFrame(columns=FEATURE_COLUMNS)
    completed = set(map(tuple, existing[["dataset", "model_slug"]].to_numpy()))
    rows = existing.to_dict("records")
    for pair in pairs:
        key = (str(pair["dataset"]), str(pair["model_slug"]))
        if key in completed:
            print(f"[skip] {key[0]} / {key[1]}")
            continue
        row = compute_pair(pair, args.vector_root, plan, device)
        rows.append(row)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows, columns=FEATURE_COLUMNS).sort_values(
            ["dataset", "model_slug"]
        ).to_csv(args.output, index=False)
        print(f"[ok] {key[0]} / {key[1]}", flush=True)
    print(f"Baseline features: {args.output}")


if __name__ == "__main__":
    main()

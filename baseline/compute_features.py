#!/usr/bin/env python3
"""Compute the dataset-level features used by the PGTuner QPP baseline."""

from __future__ import annotations

import argparse
import json
import math
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

from hnsw_cost.fingerprint import fingerprint
from hnsw_cost.vector_io import (
    load_jsonl,
    load_vectors_and_rows,
    resolve_path,
    sha256_file,
    verify_entry_files,
)

FEATURE_VALUE_COLUMNS = [
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
FEATURE_COLUMNS = [
    "dataset",
    "model_slug",
    *FEATURE_VALUE_COLUMNS,
    "manifest_sha256",
    "feature_fingerprint",
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
    """Resolve and verify the locked document and query files."""
    manifest = vector_root / str(pair["manifest_relative_path"])
    if not manifest.exists():
        raise FileNotFoundError(manifest)
    digest = sha256_file(manifest)
    if digest != str(pair["manifest_sha256"]):
        raise RuntimeError(
            f"manifest hash mismatch for {pair['dataset']}/{pair['model_slug']}"
        )
    entries = [
        entry
        for entry in load_jsonl(manifest)
        if str(entry.get("task_name", pair["dataset"])) == str(pair["dataset"])
    ]
    if not entries:
        raise RuntimeError(f"empty manifest for {pair['dataset']}/{pair['model_slug']}")
    for entry in entries:
        verify_entry_files(entry, manifest)
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

    if (
        base_vectors.ndim != 2
        or query_vectors.ndim != 2
        or base_vectors.shape[1] != query_vectors.shape[1]
        or not np.isfinite(base_vectors).all()
        or not np.isfinite(query_vectors).all()
        or batch_size <= 0
        or k <= 0
        or k > len(base_vectors) - int(exclude_aligned_self)
    ):
        raise ValueError("invalid exact-neighbor inputs")
    if exclude_aligned_self and len(base_vectors) != len(query_vectors):
        raise ValueError("aligned self-exclusion requires equal vector counts")
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
    pair: dict,
    document_dir: Path,
    query_dir: Path,
    specification: dict,
    device: str,
    feature_fingerprint: str,
) -> dict:
    """Compute every PGTuner feature for one dataset/model pair."""
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
        "manifest_sha256": str(pair["manifest_sha256"]),
        "feature_fingerprint": feature_fingerprint,
    }
    if not np.isfinite(
        [float(result[column]) for column in FEATURE_VALUE_COLUMNS]
    ).all():
        raise RuntimeError("a baseline feature is non-finite")
    return result


def main() -> None:
    """Compute and save the selected locked-pair feature rows."""
    args = parse_args()
    plan = json.loads(args.config.read_text(encoding="utf-8"))["feature_extraction"]
    device = select_device(args.device)
    import torch

    device_name = (
        torch.cuda.get_device_name(device) if device.startswith("cuda") else "cpu"
    )
    feature_fingerprint = fingerprint(
        {
            "config_sha256": sha256_file(args.config),
            "script_sha256": sha256_file(Path(__file__)),
            "vector_io_sha256": sha256_file(
                Path(__file__).resolve().parents[1] / "src/hnsw_cost/vector_io.py"
            ),
            "numpy_version": np.__version__,
            "torch_version": metadata.version("torch"),
            "cuda_version": torch.version.cuda,
            "device": device,
            "device_name": device_name,
        }
    )
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
    if args.output.exists():
        existing = pd.read_csv(args.output)
        if (
            list(existing.columns) != FEATURE_COLUMNS
            or existing.duplicated(["dataset", "model_slug"]).any()
        ):
            existing = pd.DataFrame(columns=FEATURE_COLUMNS)
    else:
        existing = pd.DataFrame(columns=FEATURE_COLUMNS)
    rows = existing.to_dict("records")
    for pair in pairs:
        key = (str(pair["dataset"]), str(pair["model_slug"]))
        document_dir, query_dir = vector_directories(pair, args.vector_root)
        matches = [
            row for row in rows if (str(row["dataset"]), str(row["model_slug"])) == key
        ]
        if matches and not args.force:
            cached = matches[0]
            if (
                str(cached["manifest_sha256"]) == str(pair["manifest_sha256"])
                and str(cached["feature_fingerprint"]) == feature_fingerprint
            ):
                print(f"[skip] {key[0]} / {key[1]}")
                continue
        rows = [
            row for row in rows if (str(row["dataset"]), str(row["model_slug"])) != key
        ]
        row = compute_pair(
            pair,
            document_dir,
            query_dir,
            plan,
            device,
            feature_fingerprint,
        )
        rows.append(row)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows, columns=FEATURE_COLUMNS).sort_values(
            ["dataset", "model_slug"]
        ).to_csv(args.output, index=False)
        print(f"[ok] {key[0]} / {key[1]}", flush=True)
    print(f"Baseline features: {args.output}")


if __name__ == "__main__":
    main()

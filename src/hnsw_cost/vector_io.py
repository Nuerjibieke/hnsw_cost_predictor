"""Load the exported embedding vectors and their aligned records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def load_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    """Yield non-empty JSON Lines records from a file."""
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def resolve_path(path: str | Path, manifest: Path | None = None) -> Path:
    """Resolve an exported vector path relative to its manifest."""
    candidate = Path(path)
    if candidate.exists():
        return candidate
    if manifest is not None and not candidate.is_absolute():
        relative = manifest.parent / candidate
        if relative.exists():
            return relative
    if manifest is not None:
        parts = candidate.parts
        positions = [
            index for index, value in enumerate(parts) if value == manifest.parent.name
        ]
        if positions:
            relocated = manifest.parent.joinpath(*parts[positions[-1] + 1 :])
            if relocated.exists():
                return relocated
    raise FileNotFoundError(path)


def load_vectors_and_rows(vector_dir: Path) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Load all vector batches and validate their aligned records."""
    vector_parts: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for meta_path in sorted(vector_dir.glob("batch_*.meta.json")):
        stem = meta_path.name[: -len(".meta.json")]
        vectors = np.load(meta_path.with_name(f"{stem}.npy"), mmap_mode="r")
        batch_rows = list(load_jsonl(meta_path.with_name(f"{stem}.jsonl")))
        if len(vectors) != len(batch_rows):
            raise ValueError(f"vector/record mismatch in {vector_dir}")
        vector_parts.append(np.asarray(vectors, dtype=np.float32))
        rows.extend(batch_rows)
    if not vector_parts:
        raise ValueError(f"no vectors found in {vector_dir}")
    vectors = np.concatenate(vector_parts, axis=0)
    return np.ascontiguousarray(vectors, dtype=np.float32), rows


def vector_shape(vector_dir: Path) -> tuple[int, int]:
    """Return the total row count and common dimension of vector batches."""
    rows = 0
    dimension: int | None = None
    for meta_path in sorted(vector_dir.glob("batch_*.meta.json")):
        stem = meta_path.name[: -len(".meta.json")]
        vectors = np.load(meta_path.with_name(f"{stem}.npy"), mmap_mode="r")
        record_count = sum(1 for _ in load_jsonl(meta_path.with_name(f"{stem}.jsonl")))
        if len(vectors) != record_count:
            raise ValueError(f"vector/record mismatch in {vector_dir}")
        if vectors.ndim != 2 or (
            dimension is not None and vectors.shape[1] != dimension
        ):
            raise ValueError(f"inconsistent vector dimensions in {vector_dir}")
        rows += len(vectors)
        dimension = int(vectors.shape[1])
    if dimension is None:
        raise ValueError(f"no vectors found in {vector_dir}")
    return rows, dimension


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Normalize each vector to unit L2 length."""
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-30)
    return np.ascontiguousarray(vectors / norms, dtype=np.float32)

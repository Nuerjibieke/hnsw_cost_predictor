"""Load the exported embedding vectors and their aligned records."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

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


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_entry_files(entry: dict[str, Any], manifest: Path) -> None:
    """Verify vector and record hashes when the manifest provides them."""
    checks = (
        ("vectors_path", "vectors_sha256"),
        ("records_path", "records_sha256"),
    )
    for path_key, hash_key in checks:
        expected = entry.get(hash_key)
        if expected is None:
            continue
        path = resolve_path(entry[path_key], manifest)
        if sha256_file(path) != expected:
            raise RuntimeError(f"file hash mismatch: {path}")


def load_vectors_and_rows(vector_dir: Path) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Load all vector batches and validate their aligned records."""
    vector_parts: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    dimension: int | None = None
    for meta_path in sorted(vector_dir.glob("batch_*.meta.json")):
        stem = meta_path.name[: -len(".meta.json")]
        vectors = np.load(meta_path.with_name(f"{stem}.npy"), mmap_mode="r")
        batch_rows = list(load_jsonl(meta_path.with_name(f"{stem}.jsonl")))
        if len(vectors) != len(batch_rows):
            raise ValueError(f"vector/record mismatch in {vector_dir}")
        if vectors.ndim != 2 or (
            dimension is not None and vectors.shape[1] != dimension
        ):
            raise ValueError(f"inconsistent vector dimensions in {vector_dir}")
        dimension = int(vectors.shape[1])
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
    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError("vectors must be a finite two-dimensional array")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms <= 1e-30):
        raise ValueError("zero-length vectors cannot be normalized")
    return np.ascontiguousarray(vectors / norms, dtype=np.float32)

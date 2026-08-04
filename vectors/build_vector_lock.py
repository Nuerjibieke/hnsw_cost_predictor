#!/usr/bin/env python3
"""Validate exported vectors and write a pipeline vector lock."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--vector-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("configs/vector_manifests.json"))
    parser.add_argument("--tasks", nargs="*")
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("mteb_config.json")
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    """Yield non-empty JSON Lines records from one file."""
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def resolve_entry_path(manifest: Path, value: str) -> Path:
    """Resolve one manifest entry relative to its run directory."""
    path = Path(value)
    return path if path.is_absolute() else manifest.parent / path


def prompt_kind(entry: dict[str, Any]) -> str | None:
    """Return query or document for one MTEB prompt entry."""
    prompt = str(entry.get("prompt_type"))
    if "PromptType.query" in prompt:
        return "query"
    if "PromptType.document" in prompt:
        return "document"
    return None


def validate_entry(manifest: Path, entry: dict[str, Any], expected_dim: int) -> int:
    """Validate one vector batch and return its row count."""
    vector_path = resolve_entry_path(manifest, entry["vectors_path"])
    record_path = resolve_entry_path(manifest, entry["records_path"])
    if not vector_path.is_file() or not record_path.is_file():
        raise FileNotFoundError(f"missing vector batch listed by {manifest}")
    if entry.get("vectors_sha256") != sha256_file(vector_path):
        raise RuntimeError(f"vector hash mismatch: {vector_path}")
    if entry.get("records_sha256") != sha256_file(record_path):
        raise RuntimeError(f"record hash mismatch: {record_path}")
    vectors = np.load(vector_path, mmap_mode="r", allow_pickle=False)
    record_count = sum(1 for _ in load_jsonl(record_path))
    if (
        vectors.ndim != 2
        or vectors.shape[1] != expected_dim
        or len(vectors) != record_count
        or len(vectors) != int(entry["num_vectors"])
        or record_count != int(entry["record_count"])
    ):
        raise RuntimeError(f"invalid vector batch: {vector_path}")
    return len(vectors)


def build_pair(
    vector_root: Path,
    task: str,
    model: dict[str, Any],
    expected_size: dict[str, int],
) -> dict[str, Any]:
    """Validate one task/model export and return its lock entry."""
    run_parent = (
        vector_root
        / "datasets"
        / task
        / "embeddings"
        / model["name"].replace("/", "__")
        / model["revision"]
    )
    manifests = sorted(run_parent.glob("*/manifest.jsonl"))
    if len(manifests) != 1:
        raise RuntimeError(
            f"expected one vector run for {task}/{model['name']}, found {len(manifests)}"
        )
    manifest = manifests[0]
    counts = {"query": 0, "document": 0}
    entry_count = 0
    for entry in load_jsonl(manifest):
        kind = prompt_kind(entry)
        if kind is None:
            continue
        if entry.get("task_name") != task:
            raise RuntimeError(f"task mismatch in {manifest}")
        counts[kind] += validate_entry(manifest, entry, int(model["embedding_dim"]))
        entry_count += 1
    if not counts["query"] or not counts["document"]:
        raise RuntimeError(f"query or document vectors are missing in {manifest}")
    expected_counts = {
        "query": int(expected_size["queries"]),
        "document": int(expected_size["documents"]),
    }
    if counts != expected_counts:
        raise RuntimeError(f"dataset size mismatch for {task}: {counts} != {expected_counts}")
    return {
        "dataset": task,
        "model_slug": model["name"].replace("/", "__"),
        "revision": model["revision"],
        "run_id": manifest.parent.name,
        "embedding_dim": int(model["embedding_dim"]),
        "num_docs": counts["document"],
        "num_queries": counts["query"],
        "manifest_entries": entry_count,
        "manifest_relative_path": str(manifest.relative_to(vector_root)),
        "manifest_sha256": sha256_file(manifest),
    }


def main() -> None:
    """Validate the selected matrix and write its stable vector lock."""
    args = parse_args()
    vector_root = args.vector_root.expanduser().resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    allowed_tasks = [*config["datasets"], *config.get("external_datasets", [])]
    tasks = args.tasks or config["datasets"]
    unknown = sorted(set(tasks) - set(allowed_tasks))
    if unknown:
        raise ValueError(f"unknown tasks: {unknown}")
    pairs = [
        build_pair(vector_root, task, model, config["dataset_sizes"][task])
        for task in tasks
        for model in config["models"]
    ]
    expected = len(tasks) * len(config["models"])
    if len(pairs) != expected:
        raise RuntimeError(f"expected {expected} vector pairs, found {len(pairs)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"pairs": pairs}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(pairs)} pairs to {args.output}")


if __name__ == "__main__":
    main()

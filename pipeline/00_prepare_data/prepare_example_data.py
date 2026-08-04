#!/usr/bin/env python3
"""Create a compact vector workload from the locked paper embeddings."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from hnsw_cost.vector_io import (
    load_jsonl,
    resolve_path,
    sha256_file,
    verify_entry_files,
)

DEFAULT_DATASETS = ["NFCorpus", "SciFact"]
DEFAULT_MODELS = [
    "McGill-NLP__LLM2Vec-Sheared-LLaMA-mntp-supervised",
    "intfloat__e5-mistral-7b-instruct",
]


def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--source-lock", type=Path, default=Path("configs/vector_manifests.json")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("example/data/vectors")
    )
    parser.add_argument(
        "--output-lock", type=Path, default=Path("example/vector_manifests.json")
    )
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--max-docs", type=int, default=512)
    parser.add_argument("--max-queries", type=int, default=32)
    return parser.parse_args()


def read_rows(path: Path, limit: int) -> list[dict]:
    """Read compact identities for the first requested vector rows."""
    rows: list[dict] = []
    for offset, row in enumerate(load_jsonl(path)):
        rows.append(
            {
                "row_index": row.get("row_index", offset),
                "id": str(row.get("id", row.get("_id", row.get("query_id", offset)))),
            }
        )
        if len(rows) == limit:
            break
    return rows


def write_rows(path: Path, rows: list[dict]) -> None:
    """Write metadata rows as JSON Lines."""
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def selected_entries(manifest: Path, dataset: str) -> list[dict]:
    """Return the query and document entries for one locked dataset."""
    entries = [
        row for row in load_jsonl(manifest) if str(row.get("task_name")) == dataset
    ]
    entries = [
        row
        for row in entries
        if "PromptType.query" in str(row.get("prompt_type"))
        or "PromptType.document" in str(row.get("prompt_type"))
    ]
    if not any("PromptType.query" in str(row["prompt_type"]) for row in entries):
        raise RuntimeError(f"query vectors missing in {manifest}")
    if not any("PromptType.document" in str(row["prompt_type"]) for row in entries):
        raise RuntimeError(f"document vectors missing in {manifest}")
    return sorted(
        entries,
        key=lambda row: (str(row["prompt_type"]), int(row.get("batch_index", 0))),
    )


def copy_entry(
    entry: dict, source_manifest: Path, target_run: Path, limit: int
) -> dict:
    """Copy the leading vectors and aligned rows for one manifest entry."""
    verify_entry_files(entry, source_manifest)
    source_vectors = resolve_path(entry["vectors_path"], source_manifest)
    source_records = resolve_path(entry["records_path"], source_manifest)
    vectors = np.load(source_vectors, mmap_mode="r")
    if vectors.ndim != 2:
        raise RuntimeError(f"invalid vectors in {source_vectors}")
    take = min(limit, len(vectors))
    rows = read_rows(source_records, take)
    if len(rows) != take:
        raise RuntimeError(f"record/vector mismatch in {source_manifest}")

    relative_dir = (
        Path(str(entry["task_name"]))
        / f"split={entry['hf_split']}"
        / f"subset={entry['hf_subset']}"
        / f"prompt_type={entry['prompt_type']}"
    )
    target_dir = target_run / relative_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    stem = f"batch_{int(entry.get('batch_index', 0)):05d}"
    vector_path = target_dir / f"{stem}.npy"
    record_path = target_dir / f"{stem}.jsonl"
    meta_path = target_dir / f"{stem}.meta.json"
    compact_vectors = np.asarray(vectors[:take])
    if not np.isfinite(compact_vectors).all():
        raise RuntimeError(f"non-finite vectors in {source_vectors}")
    np.save(vector_path, compact_vectors)
    write_rows(record_path, rows)

    output = dict(entry)
    output.update(
        num_vectors=take,
        record_count=take,
        embedding_dim=int(compact_vectors.shape[1]),
        dtype=str(compact_vectors.dtype),
        vectors_path=str(vector_path.relative_to(target_run)),
        records_path=str(record_path.relative_to(target_run)),
        vectors_sha256=sha256_file(vector_path),
        records_sha256=sha256_file(record_path),
    )
    meta_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return output


def build_pair(
    source_root: Path, output_root: Path, pair: dict, max_docs: int, max_queries: int
) -> dict:
    """Build one compact dataset/model vector export and its lock entry."""
    source_manifest = source_root / pair["manifest_relative_path"]
    if not source_manifest.exists():
        raise FileNotFoundError(source_manifest)
    if (
        hashlib.sha256(source_manifest.read_bytes()).hexdigest()
        != pair["manifest_sha256"]
    ):
        raise RuntimeError(
            f"manifest hash mismatch for {pair['dataset']}/{pair['model_slug']}"
        )
    target_run = (
        output_root
        / "embeddings"
        / pair["model_slug"]
        / pair["revision"]
        / pair["run_id"]
    )
    if target_run.exists():
        shutil.rmtree(target_run)
    target_run.mkdir(parents=True, exist_ok=True)
    output_entries = []
    remaining = {"query": max_queries, "document": max_docs}
    for entry in selected_entries(source_manifest, pair["dataset"]):
        kind = (
            "query" if "PromptType.query" in str(entry["prompt_type"]) else "document"
        )
        if remaining[kind] <= 0:
            continue
        copied = copy_entry(entry, source_manifest, target_run, remaining[kind])
        output_entries.append(copied)
        remaining[kind] -= int(copied["num_vectors"])

    manifest = target_run / "manifest.jsonl"
    write_rows(manifest, output_entries)
    num_queries = sum(
        int(row["num_vectors"])
        for row in output_entries
        if "PromptType.query" in str(row["prompt_type"])
    )
    num_docs = sum(
        int(row["num_vectors"])
        for row in output_entries
        if "PromptType.document" in str(row["prompt_type"])
    )
    if not num_queries or not num_docs:
        raise RuntimeError(
            f"empty compact vectors for {pair['dataset']}/{pair['model_slug']}"
        )
    return {
        "dataset": pair["dataset"],
        "model_slug": pair["model_slug"],
        "revision": pair["revision"],
        "run_id": pair["run_id"],
        "embedding_dim": int(pair["embedding_dim"]),
        "num_docs": num_docs,
        "num_queries": num_queries,
        "manifest_entries": len(output_entries),
        "manifest_relative_path": str(manifest.relative_to(output_root)),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }


def main() -> None:
    """Create all selected compact vector exports and their lock file."""
    args = parse_args()
    args.source_root = args.source_root.expanduser().resolve()
    args.source_lock = args.source_lock.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.output_lock = args.output_lock.expanduser().resolve()
    if (
        args.source_root == args.output_root
        or args.source_root in args.output_root.parents
        or args.output_root in args.source_root.parents
    ):
        raise RuntimeError("source and compact vector roots must be separate")
    if args.max_docs < 40:
        raise SystemExit("--max-docs must be at least 40 for the query features")
    if args.max_queries <= 0:
        raise SystemExit("--max-queries must be positive")
    source_pairs = json.loads(args.source_lock.read_text(encoding="utf-8"))["pairs"]
    selected = [
        row
        for row in source_pairs
        if row["dataset"] in set(args.datasets)
        and row["model_slug"] in set(args.models)
    ]
    expected = len(set(args.datasets)) * len(set(args.models))
    if len(selected) != expected:
        raise RuntimeError(f"expected {expected} selected pairs, found {len(selected)}")
    output_pairs = [
        build_pair(
            args.source_root, args.output_root, row, args.max_docs, args.max_queries
        )
        for row in selected
    ]
    args.output_lock.parent.mkdir(parents=True, exist_ok=True)
    args.output_lock.write_text(
        json.dumps({"pairs": output_pairs}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"pairs": len(output_pairs), "output": str(args.output_lock)}, indent=2
        )
    )


if __name__ == "__main__":
    main()

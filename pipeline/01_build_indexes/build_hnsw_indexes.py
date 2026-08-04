#!/usr/bin/env python3
"""Build the fixed grid of FAISS HNSW indexes used by the paper."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hnsw_cost.vector_io import (
    l2_normalize,
    load_jsonl,
    load_vectors_and_rows,
    resolve_path,
    verify_entry_files,
    vector_shape,
)


@dataclass(frozen=True)
class VectorCollection:
    """Describe one locked document/query embedding collection."""

    model_slug: str
    revision: str
    run_id: str
    task_name: str
    split: str
    subset: str
    doc_dir: Path
    query_dir: Path
    num_docs: int
    num_queries: int
    embedding_dim: int
    manifest_sha256: str


def parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--vector-root", type=Path, default=Path("data/vectors"))
    parser.add_argument(
        "--lock", type=Path, default=Path("configs/vector_manifests.json")
    )
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/indexes"))
    parser.add_argument("--grid", type=Path, default=Path("configs/hnsw_grid.json"))
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--tasks", nargs="*")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    if args.threads <= 0:
        raise SystemExit("--threads must be positive")
    return args


def sanitize(value: object) -> str:
    """Convert a value to a safe path component."""
    text = str(value).strip().replace("/", "__")
    return re.sub(r"[^A-Za-z0-9._=-]+", "_", text) or "none"


def manifest_dir(entries: list[dict], kind: str, manifest: Path) -> Path:
    """Return the single batch directory for one prompt type."""
    paths = {
        resolve_path(entry["vectors_path"], manifest).parent
        for entry in entries
        if f"PromptType.{kind}" in str(entry.get("prompt_type"))
    }
    if len(paths) != 1:
        raise RuntimeError(
            f"expected one {kind} directory in {manifest}, found {len(paths)}"
        )
    return paths.pop()


def load_collections(args: argparse.Namespace) -> list[VectorCollection]:
    """Load and validate the selected collections from the vector lock."""
    pairs = json.loads(args.lock.read_text(encoding="utf-8"))["pairs"]
    collections: list[VectorCollection] = []
    for pair in pairs:
        dataset = str(pair["dataset"])
        model_slug = str(pair["model_slug"])
        if args.tasks and dataset not in args.tasks:
            continue
        if args.models and not any(
            value.lower() in model_slug.lower() for value in args.models
        ):
            continue
        manifest = args.vector_root / pair["manifest_relative_path"]
        if not manifest.exists():
            raise FileNotFoundError(manifest)
        if hashlib.sha256(manifest.read_bytes()).hexdigest() != pair["manifest_sha256"]:
            raise RuntimeError(f"manifest hash mismatch for {dataset}/{model_slug}")
        entries = [
            entry
            for entry in load_jsonl(manifest)
            if str(entry.get("task_name", dataset)) == dataset
        ]
        for entry in entries:
            verify_entry_files(entry, manifest)
        query_dir = manifest_dir(entries, "query", manifest)
        doc_dir = manifest_dir(entries, "document", manifest)
        reference = entries[0]
        collections.append(
            VectorCollection(
                model_slug=model_slug,
                revision=str(pair["revision"]),
                run_id=str(pair["run_id"]),
                task_name=dataset,
                split=str(reference.get("hf_split") or "default"),
                subset=str(reference.get("hf_subset") or "default"),
                doc_dir=doc_dir,
                query_dir=query_dir,
                num_docs=int(pair["num_docs"]),
                num_queries=int(pair["num_queries"]),
                embedding_dim=int(pair["embedding_dim"]),
                manifest_sha256=str(pair["manifest_sha256"]),
            )
        )
    if not collections:
        raise RuntimeError("no locked vector collections match the filters")
    return collections


def output_dir(
    root: Path, collection: VectorCollection, ef_construction: int, m_value: int
) -> Path:
    """Return the deterministic directory for one HNSW index."""
    return (
        root
        / "experiments"
        / sanitize(collection.task_name)
        / f"split={sanitize(collection.split)}"
        / f"subset={sanitize(collection.subset)}"
        / collection.model_slug
        / collection.revision
        / collection.run_id
        / f"efc={ef_construction}_M={m_value}"
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write one formatted JSON object."""
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    """Build every selected index exactly once."""
    args = parse_args()
    grid = json.loads(args.grid.read_text(encoding="utf-8"))
    metric_name = str(grid["metric"])
    metric = "ip" if metric_name == "inner_product" else metric_name
    normalize = bool(grid["normalize_embeddings"])
    top_k = int(grid["top_k"])
    pairs = [
        (int(efc), int(m))
        for efc in grid["ef_construction_values"]
        for m in grid["M_values"]
    ]
    if (
        metric not in {"ip", "l2"}
        or top_k <= 0
        or any(a <= 0 or b <= 0 for a, b in pairs)
    ):
        raise RuntimeError("invalid HNSW grid")

    import faiss

    faiss.omp_set_num_threads(args.threads)
    collections = load_collections(args)
    print(f"Found {len(collections)} locked vector collection(s).")
    for collection in collections:
        docs, _ = load_vectors_and_rows(collection.doc_dir)
        num_queries, query_dim = vector_shape(collection.query_dir)
        if (
            docs.shape != (collection.num_docs, collection.embedding_dim)
            or (num_queries, query_dim)
            != (collection.num_queries, collection.embedding_dim)
            or top_k > len(docs)
        ):
            raise RuntimeError(
                f"invalid vector shapes for {collection.task_name}/{collection.model_slug}"
            )
        if normalize:
            docs = l2_normalize(docs)
        for ef_construction, m_value in pairs:
            target = output_dir(args.output_root, collection, ef_construction, m_value)
            summary_path = target / "summary.json"
            index_path = target / "docs_hnsw.faiss"
            if args.skip_existing and summary_path.exists() and index_path.exists():
                prior = json.loads(summary_path.read_text(encoding="utf-8"))
                reusable = (
                    prior.get("manifest_sha256") == collection.manifest_sha256
                    and prior.get("metric") == metric
                    and prior.get("normalize") == normalize
                    and prior.get("k") == top_k
                    and prior.get("ef_construction") == ef_construction
                    and prior.get("M") == m_value
                    and prior.get("num_docs") == collection.num_docs
                    and prior.get("num_queries") == collection.num_queries
                    and prior.get("embedding_dim") == collection.embedding_dim
                )
                if reusable:
                    print(
                        f"[skip] {collection.task_name} {collection.model_slug} efc={ef_construction} M={m_value}"
                    )
                    continue
            target.mkdir(parents=True, exist_ok=True)
            started = time.perf_counter()
            faiss_metric = (
                faiss.METRIC_INNER_PRODUCT if metric == "ip" else faiss.METRIC_L2
            )
            index = faiss.IndexHNSWFlat(docs.shape[1], m_value, faiss_metric)
            index.hnsw.efConstruction = ef_construction
            index.add(docs)
            build_seconds = time.perf_counter() - started
            faiss.write_index(index, str(index_path))
            write_json(
                summary_path,
                {
                    "attempt_dir": str(target),
                    "index_path": str(index_path),
                    "query_dir": str(collection.query_dir),
                    "task_name": collection.task_name,
                    "split": collection.split,
                    "subset": collection.subset,
                    "model_slug": collection.model_slug,
                    "revision": collection.revision,
                    "run_id": collection.run_id,
                    "metric": metric,
                    "normalize": normalize,
                    "k": top_k,
                    "ef_construction": ef_construction,
                    "M": m_value,
                    "num_docs": len(docs),
                    "num_queries": num_queries,
                    "embedding_dim": int(docs.shape[1]),
                    "build_seconds": build_seconds,
                    "manifest_sha256": collection.manifest_sha256,
                },
            )
            print(
                f"[ok] {collection.task_name} {collection.model_slug} efc={ef_construction} M={m_value}"
            )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Measure per-query HNSW distance computations for the fixed search grid."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hnsw_cost.vector_io import l2_normalize, load_vectors_and_rows


@dataclass(frozen=True)
class Attempt:
    """Describe one saved HNSW index and its query vectors."""

    summary_path: Path
    index_path: Path
    query_dir: Path
    task_name: str
    split: str
    subset: str
    model_slug: str
    revision: str
    run_id: str
    normalize: bool
    k: int
    ef_construction: int
    m_value: int
    num_docs: int
    num_queries: int
    embedding_dim: int
    manifest_sha256: str


def parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--hnsw-root", type=Path, default=Path("artifacts/indexes"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("artifacts/query_costs")
    )
    parser.add_argument("--grid", type=Path, default=Path("configs/hnsw_grid.json"))
    parser.add_argument("--tasks", nargs="*")
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--faiss-threads", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.workers <= 0 or args.faiss_threads <= 0:
        raise SystemExit("worker and thread counts must be positive")
    return args


def sanitize(value: object) -> str:
    """Convert a value to a filesystem-safe component."""
    text = str(value if value is not None else "none")
    return "".join(ch if ch.isalnum() or ch in "._=-" else "_" for ch in text) or "none"


def load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def parse_attempt(summary_path: Path) -> Attempt | None:
    """Parse one complete HNSW build summary."""
    try:
        payload = load_json(summary_path)
        attempt = Attempt(
            summary_path=summary_path,
            index_path=Path(payload["index_path"]),
            query_dir=Path(payload["query_dir"]),
            task_name=str(payload["task_name"]),
            split=str(payload["split"]),
            subset=str(payload["subset"]),
            model_slug=str(payload["model_slug"]),
            revision=str(payload["revision"]),
            run_id=str(payload["run_id"]),
            normalize=bool(payload["normalize"]),
            k=int(payload["k"]),
            ef_construction=int(payload["ef_construction"]),
            m_value=int(payload["M"]),
            num_docs=int(payload["num_docs"]),
            num_queries=int(payload["num_queries"]),
            embedding_dim=int(payload["embedding_dim"]),
            manifest_sha256=str(payload["manifest_sha256"]),
        )
    except (KeyError, OSError, ValueError, json.JSONDecodeError):
        return None
    return (
        attempt if attempt.index_path.exists() and attempt.query_dir.exists() else None
    )


def discover_attempts(args: argparse.Namespace) -> list[Attempt]:
    """Discover saved indexes that pass optional task and model filters."""
    attempts = [
        attempt
        for path in sorted((args.hnsw_root / "experiments").rglob("summary.json"))
        if (attempt := parse_attempt(path)) is not None
    ]
    if args.tasks:
        attempts = [attempt for attempt in attempts if attempt.task_name in args.tasks]
    if args.models:
        attempts = [
            attempt
            for attempt in attempts
            if any(value.lower() in attempt.model_slug.lower() for value in args.models)
        ]
    return sorted(
        attempts,
        key=lambda item: (
            item.task_name,
            item.model_slug,
            item.ef_construction,
            item.m_value,
        ),
    )


def output_dir(root: Path, attempt: Attempt) -> Path:
    """Return the measurement directory for one HNSW index."""
    return (
        root
        / "query_curves"
        / sanitize(attempt.task_name)
        / f"split={sanitize(attempt.split)}"
        / f"subset={sanitize(attempt.subset)}"
        / sanitize(attempt.model_slug)
        / sanitize(attempt.revision)
        / sanitize(attempt.run_id)
        / f"efc={attempt.ef_construction}_M={attempt.m_value}"
    )


def record_id(rows: list[dict[str, Any]], offset: int) -> str:
    """Return the stable identifier for one query vector."""
    row = rows[offset]
    return str(row.get("id", row.get("_id", row.get("query_id", offset))))


def measure_attempt(
    attempt_data: dict[str, Any],
    output_root: str,
    ef_values: list[int],
    faiss_threads: int,
    force: bool,
) -> dict[str, Any]:
    """Measure every query and efSearch value for one saved index."""
    attempt = Attempt(**attempt_data)
    target = output_dir(Path(output_root), attempt)
    curves_path = target / "query_curve.jsonl.gz"
    meta_path = target / "meta.json"
    done_path = target / "_SUCCESS"
    expected_rows = attempt.num_queries * len(ef_values)
    grid_text = ",".join(map(str, ef_values))
    if not force and curves_path.exists() and meta_path.exists() and done_path.exists():
        prior = load_json(meta_path)
        if (
            prior.get("curve_rows") == expected_rows
            and prior.get("efSearch_grid") == grid_text
            and prior.get("manifest_sha256") == attempt.manifest_sha256
        ):
            return prior

    import faiss

    faiss.omp_set_num_threads(faiss_threads)
    queries, query_rows = load_vectors_and_rows(attempt.query_dir)
    if len(queries) != attempt.num_queries or queries.shape[1] != attempt.embedding_dim:
        raise RuntimeError(f"query shape mismatch for {attempt.summary_path}")
    if attempt.normalize:
        queries = l2_normalize(queries)
    index = faiss.read_index(str(attempt.index_path))
    if index.ntotal != attempt.num_docs or index.d != attempt.embedding_dim:
        raise RuntimeError(f"index shape mismatch for {attempt.summary_path}")

    target.mkdir(parents=True, exist_ok=True)
    row_count = 0
    with gzip.open(curves_path, "wt", encoding="utf-8") as handle:
        for ef_search in ef_values:
            index.hnsw.efSearch = ef_search
            for offset in range(len(queries)):
                query = np.ascontiguousarray(
                    queries[offset : offset + 1], dtype=np.float32
                )
                faiss.cvar.hnsw_stats.reset()
                index.search(query, attempt.k)
                row = {
                    "query_offset": offset,
                    "query_id": record_id(query_rows, offset),
                    "M": attempt.m_value,
                    "efConstruction": attempt.ef_construction,
                    "efSearch": ef_search,
                    "distance_computations": int(faiss.cvar.hnsw_stats.ndis),
                }
                handle.write(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
                row_count += 1
    if row_count != expected_rows:
        raise RuntimeError(f"measured {row_count} rows, expected {expected_rows}")
    meta = {
        "task_name": attempt.task_name,
        "split": attempt.split,
        "subset": attempt.subset,
        "model_slug": attempt.model_slug,
        "revision": attempt.revision,
        "run_id": attempt.run_id,
        "M": attempt.m_value,
        "efConstruction": attempt.ef_construction,
        "num_docs": attempt.num_docs,
        "num_queries": attempt.num_queries,
        "efSearch_grid": grid_text,
        "curve_rows": row_count,
        "manifest_sha256": attempt.manifest_sha256,
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    done_path.write_text("ok\n", encoding="utf-8")
    return meta


def main() -> None:
    """Measure all discovered indexes and stop if any worker fails."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    args = parse_args()
    grid = load_json(args.grid)
    ef_values = sorted({int(value) for value in grid["ef_search_values"]})
    if not ef_values or any(value <= 0 for value in ef_values):
        raise RuntimeError("invalid efSearch grid")
    attempts = discover_attempts(args)
    if not attempts:
        raise RuntimeError("no matching HNSW indexes found")
    failures: list[str] = []
    rows = 0
    if args.workers == 1:
        for attempt in attempts:
            result = measure_attempt(
                asdict(attempt),
                str(args.output_root),
                ef_values,
                args.faiss_threads,
                args.force,
            )
            rows += int(result["curve_rows"])
            print(
                f"[ok] {attempt.task_name} {attempt.model_slug} efc={attempt.ef_construction} M={attempt.m_value}"
            )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    measure_attempt,
                    asdict(attempt),
                    str(args.output_root),
                    ef_values,
                    args.faiss_threads,
                    args.force,
                ): attempt
                for attempt in attempts
            }
            for future in as_completed(futures):
                attempt = futures[future]
                try:
                    result = future.result()
                    rows += int(result["curve_rows"])
                    print(
                        f"[ok] {attempt.task_name} {attempt.model_slug} efc={attempt.ef_construction} M={attempt.m_value}"
                    )
                except Exception as error:
                    failures.append(f"{attempt.summary_path}: {error!r}")
    if failures:
        raise RuntimeError("\n".join(failures))
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "summary.json").write_text(
        json.dumps({"indexes": len(attempts), "rows": rows}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Collect native per-query distance-comparison labels from DiskANN Vamana."""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hnsw_cost.vector_io import sha256_file

from common import (
    locked_pairs,
    pair_embeddings,
    pair_fingerprint,
    pair_output,
    valid_output,
)


DISKANN_COMMIT = "78256bbab4685e1774e78d331e081a153be26823"
COUNTER_PATCH = Path(__file__).with_name("per_query_search_stats.patch")


def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--diskann-root", type=Path, required=True)
    parser.add_argument("--vector-root", type=Path, required=True)
    parser.add_argument("--lock", type=Path, default=Path("configs/vector_manifests.json"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--build-threads", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def check_diskann(root: Path) -> tuple[Path, Path]:
    """Require the pinned and patched DiskANN build and search programs."""
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    if commit != DISKANN_COMMIT:
        raise RuntimeError("DiskANN revision does not match the paper")
    patch_check = subprocess.run(
        ["git", "apply", "--reverse", "--check", str(COUNTER_PATCH)],
        cwd=root,
        text=True,
        capture_output=True,
    )
    if patch_check.returncode != 0:
        raise RuntimeError("the DiskANN comparison-counter patch is not applied")
    build = root / "build" / "apps" / "build_memory_index"
    search = root / "build" / "apps" / "search_memory_index"
    if not build.is_file() or not search.is_file():
        raise FileNotFoundError("DiskANN build/search programs are missing")
    return build, search


def write_diskann_matrix(path: Path, values: np.ndarray) -> None:
    """Write a dense matrix in DiskANN binary input format."""
    values = np.ascontiguousarray(values)
    if values.ndim != 2:
        raise ValueError("DiskANN input must be a matrix")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.asarray(values.shape, dtype=np.uint32).tofile(handle)
        values.tofile(handle)


def read_diskann_matrix(path: Path, dtype: np.dtype[Any]) -> np.ndarray:
    """Read one dense matrix from DiskANN binary output format."""
    with path.open("rb") as handle:
        shape = np.fromfile(handle, dtype=np.uint32, count=2)
        values = np.fromfile(handle, dtype=dtype)
    if len(shape) != 2 or values.size != int(shape[0]) * int(shape[1]):
        raise RuntimeError(f"invalid DiskANN matrix: {path}")
    return values.reshape(int(shape[0]), int(shape[1]))


def run_command(command: list[str], threads: int) -> None:
    """Run one DiskANN command with fixed CPU thread settings."""
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": str(threads),
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    subprocess.run(command, env=environment, check=True)


def collect_pair(
    build_binary: Path,
    search_binary: Path,
    queries: np.ndarray,
    docs: np.ndarray,
    pair: dict,
    work_root: Path,
    build_threads: int,
    input_fingerprint: str,
) -> pd.DataFrame:
    """Build four Vamana indexes and collect all 16 query-setting workloads."""
    binary_root = work_root / "vectors"
    index_root = work_root / "indexes"
    result_root = work_root / "search"
    docs_path = binary_root / "documents.bin"
    queries_path = binary_root / "queries.bin"
    write_diskann_matrix(docs_path, docs)
    write_diskann_matrix(queries_path, queries)
    rows = []
    for r_value in (32, 64):
        for build_l in (64, 128):
            key = f"R={r_value}_Lbuild={build_l}"
            index_prefix = index_root / key / "vamana"
            index_prefix.parent.mkdir(parents=True, exist_ok=True)
            run_command(
                [
                    str(build_binary),
                    "--data_type",
                    "float",
                    "--dist_fn",
                    "cosine",
                    "--data_path",
                    str(docs_path),
                    "--index_path_prefix",
                    str(index_prefix),
                    "-R",
                    str(r_value),
                    "-L",
                    str(build_l),
                    "--alpha",
                    "1.2",
                    "-T",
                    str(build_threads),
                ],
                build_threads,
            )
            result_prefix = result_root / key / "result"
            result_prefix.parent.mkdir(parents=True, exist_ok=True)
            search_values = (16, 64, 256, 512)
            run_command(
                [
                    str(search_binary),
                    "--data_type",
                    "float",
                    "--dist_fn",
                    "cosine",
                    "--index_path_prefix",
                    str(index_prefix),
                    "--result_path",
                    str(result_prefix),
                    "--query_file",
                    str(queries_path),
                    "-K",
                    "10",
                    "-L",
                    *[str(value) for value in search_values],
                    "-T",
                    "1",
                ],
                1,
            )
            for search_l in search_values:
                count_path = Path(f"{result_prefix}_{search_l}_cmps_uint32.bin")
                counts = read_diskann_matrix(count_path, np.uint32)
                if counts.shape != (len(queries), 1) or np.any(counts[:, 0] <= 0):
                    raise RuntimeError(f"invalid native Vamana counters for {key}")
                for query_offset, count in enumerate(counts[:, 0]):
                    rows.append(
                        {
                            "dataset": pair["dataset"],
                            "model_slug": pair["model_slug"],
                            "query_offset": query_offset,
                            "vamana_R": r_value,
                            "vamana_Lbuild": build_l,
                            "vamana_Lsearch": search_l,
                            "num_docs": len(docs),
                            "distance_computations": int(count),
                            "true_log_cost": math.log1p(int(count)),
                            "alpha": 1.2,
                            "counter_valid": 1,
                            "diskann_commit": DISKANN_COMMIT,
                            "input_fingerprint": input_fingerprint,
                        }
                    )
    frame = pd.DataFrame(rows)
    expected = len(queries) * 16
    if len(frame) != expected:
        raise RuntimeError(f"expected {expected} Vamana rows, found {len(frame)}")
    return frame


def main() -> None:
    """Collect the assigned Vamana pairs and write one label shard per pair."""
    args = parse_args()
    if (
        args.num_workers <= 0
        or not 0 <= args.worker_id < args.num_workers
        or args.build_threads <= 0
    ):
        raise SystemExit("invalid Vamana worker or thread setting")
    build_binary, search_binary = check_diskann(args.diskann_root)
    pairs = locked_pairs(args.lock)[args.worker_id :: args.num_workers]
    for pair in pairs:
        output = pair_output(args.output_root, pair)
        protocol = {
            "index": "Vamana",
            "diskann_commit": DISKANN_COMMIT,
            "counter_patch_sha256": sha256_file(COUNTER_PATCH),
            "collector_sha256": sha256_file(Path(__file__)),
            "R": [32, 64],
            "Lbuild": [64, 128],
            "Lsearch": [16, 64, 256, 512],
            "alpha": 1.2,
            "build_threads": args.build_threads,
        }
        input_fingerprint = pair_fingerprint(pair, protocol)
        expected_rows = int(pair["num_queries"]) * 16
        if (
            output.exists()
            and not args.force
            and valid_output(
                output,
                input_fingerprint,
                expected_rows,
                ["query_offset", "vamana_R", "vamana_Lbuild", "vamana_Lsearch"],
            )
        ):
            continue
        work = output.parent / ".work"
        if work.exists():
            shutil.rmtree(work)
        try:
            queries, docs = pair_embeddings(args.vector_root, pair)
            frame = collect_pair(
                build_binary,
                search_binary,
                queries,
                docs,
                pair,
                work,
                args.build_threads,
                input_fingerprint,
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(output, index=False, compression="zstd")
        finally:
            if work.exists():
                shutil.rmtree(work)
        print(f"[ok] Vamana {pair['dataset']} / {pair['model_slug']}", flush=True)


if __name__ == "__main__":
    main()

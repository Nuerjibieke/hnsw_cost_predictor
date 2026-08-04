#!/usr/bin/env python3
"""Collect validated per-query distance-computation labels from FAISS NSG."""

from __future__ import annotations

import argparse
import math
import subprocess
from pathlib import Path

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


FAISS_TAG = "v1.10.0"
COUNTER_PATCH = Path(__file__).with_name("faiss_nsg_stats.patch")


def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--faiss-root", type=Path, required=True)
    parser.add_argument("--vector-root", type=Path, required=True)
    parser.add_argument("--lock", type=Path, default=Path("configs/vector_manifests.json"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def check_faiss(root: Path) -> None:
    """Require the tagged FAISS source and native NSG counter patch."""
    import faiss

    tag = subprocess.run(
        ["git", "describe", "--tags", "--exact-match", "HEAD"],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    if tag != FAISS_TAG or faiss.__version__ != "1.10.0":
        raise RuntimeError("FAISS source and runtime must both be version 1.10.0")
    patch_check = subprocess.run(
        ["git", "apply", "--reverse", "--check", str(COUNTER_PATCH)],
        cwd=root,
        text=True,
        capture_output=True,
    )
    if patch_check.returncode != 0 or not hasattr(faiss.cvar, "nsg_stats"):
        raise RuntimeError("the native FAISS NSG counter patch is not active")
    faiss.cvar.nsg_stats.reset()


def collect_pair(
    queries: np.ndarray,
    docs: np.ndarray,
    pair: dict,
    threads: int,
    input_fingerprint: str,
) -> pd.DataFrame:
    """Build two NSG indexes and collect all eight query-setting workloads."""
    import faiss

    faiss.omp_set_num_threads(threads)
    configurations = {16: (48, 116), 32: (64, 132)}
    search_values = [16, 32, 64, 128]
    rows = []
    for r_value, (expected_l, expected_c) in configurations.items():
        index = faiss.IndexNSGFlat(docs.shape[1], r_value, faiss.METRIC_INNER_PRODUCT)
        index.GK = 64
        index.add(docs)
        if int(index.nsg.L) != expected_l or int(index.nsg.C) != expected_c:
            raise RuntimeError("FAISS NSG construction settings do not match the paper")
        for search_l in search_values:
            index.nsg.search_L = search_l
            for query_offset, query in enumerate(queries):
                query = np.ascontiguousarray(query, dtype=np.float32)
                faiss.cvar.nsg_stats.reset()
                faiss_scores, faiss_ids = index.search(query.reshape(1, -1), 10)
                count = int(faiss.cvar.nsg_stats.ndis)
                if (
                    count <= 0
                    or np.any(np.asarray(faiss_ids[0]) < 0)
                    or not np.isfinite(np.asarray(faiss_scores[0])).all()
                ):
                    raise RuntimeError(f"invalid native NSG result at query {query_offset}")
                rows.append(
                    {
                        "dataset": pair["dataset"],
                        "model_slug": pair["model_slug"],
                        "query_offset": query_offset,
                        "nsg_R": r_value,
                        "nsg_L": expected_l,
                        "nsg_C": expected_c,
                        "nsg_GK": 64,
                        "search_L": search_l,
                        "num_docs": len(docs),
                        "distance_computations": count,
                        "true_log_cost": math.log1p(count),
                        "counter_valid": 1,
                        "faiss_version": faiss.__version__,
                        "input_fingerprint": input_fingerprint,
                    }
                )
    frame = pd.DataFrame(rows)
    expected = len(queries) * 8
    if len(frame) != expected:
        raise RuntimeError(f"expected {expected} NSG rows, found {len(frame)}")
    return frame


def main() -> None:
    """Collect the assigned NSG pairs and write one label shard per pair."""
    args = parse_args()
    if args.num_workers <= 0 or not 0 <= args.worker_id < args.num_workers or args.threads <= 0:
        raise SystemExit("invalid NSG worker or thread setting")
    check_faiss(args.faiss_root)
    pairs = locked_pairs(args.lock)[args.worker_id :: args.num_workers]
    for pair in pairs:
        output = pair_output(args.output_root, pair)
        protocol = {
            "index": "NSG",
            "faiss_tag": FAISS_TAG,
            "counter_patch_sha256": sha256_file(COUNTER_PATCH),
            "collector_sha256": sha256_file(Path(__file__)),
            "R": [16, 32],
            "GK": 64,
            "search_L": [16, 32, 64, 128],
            "threads": args.threads,
        }
        input_fingerprint = pair_fingerprint(pair, protocol)
        expected_rows = int(pair["num_queries"]) * 8
        if (
            output.exists()
            and not args.force
            and valid_output(
                output,
                input_fingerprint,
                expected_rows,
                ["query_offset", "nsg_R", "search_L"],
            )
        ):
            continue
        queries, docs = pair_embeddings(args.vector_root, pair)
        frame = collect_pair(
            queries, docs, pair, args.threads, input_fingerprint
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(output, index=False, compression="zstd")
        print(f"[ok] NSG {pair['dataset']} / {pair['model_slug']}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compute the canonical 16 query-local features from locked vector inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from hnsw_cost.features import (
    QUERY_FEATURES,
    compute_final_query_features,
    load_embeddings,
)


def atomic_json(path: Path, payload: object) -> None:
    """Write a JSON file atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    """Run the command-line workflow."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--vector-root", type=Path, required=True)
    parser.add_argument(
        "--lock", type=Path, default=Path("configs/vector_manifests.json")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("artifacts/query_features")
    )
    parser.add_argument("--datasets", nargs="*")
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--query-block-size", type=int, default=16)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.query_block_size <= 0:
        raise SystemExit("--query-block-size must be positive")

    lock = json.loads(args.lock.read_text())
    selected = [
        row
        for row in lock["pairs"]
        if (not args.datasets or row["dataset"] in args.datasets)
        and (not args.models or row["model_slug"] in args.models)
    ]
    if not selected:
        raise RuntimeError("no locked dataset/model pair matches the filters")

    statuses = []
    for index, row in enumerate(selected, 1):
        dataset = row["dataset"]
        model_slug = row["model_slug"]
        safe_pair = f"{dataset}__{model_slug.replace('/', '__')}"
        output = args.output_root / f"{safe_pair}.parquet"
        status_path = args.output_root / "status" / f"{safe_pair}.json"
        if output.exists() and status_path.exists() and not args.force:
            prior = json.loads(status_path.read_text())
            if (
                prior.get("status") == "pass"
                and prior.get("manifest_sha256") == row["manifest_sha256"]
            ):
                statuses.append(prior)
                continue
        queries, docs, manifest = load_embeddings(
            args.vector_root,
            dataset,
            model_slug,
            revision=row["revision"],
            run_id=row["run_id"],
        )
        actual_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
        if actual_hash != row["manifest_sha256"]:
            raise RuntimeError(f"manifest hash mismatch for {dataset}/{model_slug}")
        if len(queries) != row["num_queries"] or len(docs) != row["num_docs"]:
            raise RuntimeError(
                f"vector cardinality mismatch for {dataset}/{model_slug}"
            )
        frame = compute_final_query_features(
            queries,
            docs,
            np.arange(len(queries), dtype=np.int64),
            query_block_size=args.query_block_size,
        )
        if list(frame.columns) != ["query_offset", *QUERY_FEATURES]:
            raise RuntimeError("feature schema mismatch")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
        frame.to_parquet(temporary, index=False, compression="zstd")
        temporary.replace(output)
        missing = {
            name: int(np.isnan(frame[name].to_numpy(float)).sum())
            for name in QUERY_FEATURES
        }
        infinite = {
            name: int(np.isinf(frame[name].to_numpy(float)).sum())
            for name in QUERY_FEATURES
        }
        if any(infinite.values()):
            raise RuntimeError(f"infinite feature values for {dataset}/{model_slug}")
        status = {
            "status": "pass",
            "dataset": dataset,
            "model_slug": model_slug,
            "revision": row["revision"],
            "run_id": row["run_id"],
            "manifest_sha256": actual_hash,
            "num_queries": len(queries),
            "num_docs": len(docs),
            "embedding_dim": int(docs.shape[1]),
            "feature_count": len(QUERY_FEATURES),
            "missing_values": missing,
            "infinite_values": infinite,
            "output": output.name,
        }
        atomic_json(status_path, status)
        statuses.append(status)
        print(f"[{index}/{len(selected)}] {dataset}/{model_slug}", flush=True)
    summary = {
        "status": "pass",
        "pairs": len(statuses),
        "queries": sum(int(row["num_queries"]) for row in statuses),
        "missing_values": {
            name: sum(int(row["missing_values"][name]) for row in statuses)
            for name in QUERY_FEATURES
        },
        "infinite_values": {
            name: sum(int(row["infinite_values"][name]) for row in statuses)
            for name in QUERY_FEATURES
        },
    }
    atomic_json(args.output_root / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

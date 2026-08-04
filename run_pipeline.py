#!/usr/bin/env python3
"""Run the complete HNSW cost-prediction pipeline on exported vectors."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vector-root", type=Path, default=REPOSITORY / "example/data/vectors"
    )
    parser.add_argument(
        "--lock", type=Path, default=REPOSITORY / "example/vector_manifests.json"
    )
    parser.add_argument(
        "--output-root", type=Path, default=REPOSITORY / "example/output"
    )
    parser.add_argument(
        "--results-root", type=Path, default=REPOSITORY / "example/results"
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--faiss-threads", type=int, default=1)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.workers <= 0 or args.faiss_threads <= 0:
        raise SystemExit("worker and thread counts must be positive")
    return args


def run(command: list[str], environment: dict[str, str]) -> None:
    """Run one pipeline stage and stop if it fails."""
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPOSITORY, env=environment, check=True)


def remove_generated_path(path: Path) -> None:
    """Remove one file or directory created by an earlier pipeline run."""
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def main() -> None:
    """Execute every stage and write the two reported result tables."""
    args = parse_args()
    args.vector_root = args.vector_root.expanduser().resolve()
    args.lock = args.lock.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.results_root = args.results_root.expanduser().resolve()
    if not args.vector_root.exists() or not args.lock.exists():
        raise FileNotFoundError("vector root or lock file is missing")
    python = sys.executable
    environment = os.environ.copy()
    source = str(REPOSITORY / "src")
    environment["PYTHONPATH"] = source + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )

    indexes = args.output_root / "indexes"
    costs = args.output_root / "query_costs"
    features = args.output_root / "query_features"
    training_table = args.output_root / "training_table.parquet"
    training = args.output_root / "joint_holdout"
    for path in (indexes, costs, features, training, training_table):
        remove_generated_path(path)
    remove_generated_path(args.output_root / "training_table_summary.json")
    remove_generated_path(args.results_root / "summary.csv")
    remove_generated_path(args.results_root / "kendall.csv")
    args.results_root.mkdir(parents=True, exist_ok=True)

    run(
        [
            python,
            "pipeline/01_build_indexes/build_hnsw_indexes.py",
            "--vector-root",
            str(args.vector_root),
            "--output-root",
            str(indexes),
            "--lock",
            str(args.lock),
            "--threads",
            str(args.faiss_threads),
        ],
        environment,
    )
    run(
        [
            python,
            "pipeline/02_measure_query_costs/measure_query_costs.py",
            "--hnsw-root",
            str(indexes),
            "--output-root",
            str(costs),
            "--workers",
            str(args.workers),
            "--faiss-threads",
            str(args.faiss_threads),
        ],
        environment,
    )
    run(
        [
            python,
            "pipeline/03_extract_query_features/compute_query_features.py",
            "--vector-root",
            str(args.vector_root),
            "--lock",
            str(args.lock),
            "--output-root",
            str(features),
        ],
        environment,
    )
    run(
        [
            python,
            "pipeline/04_build_training_table/build_training_table.py",
            "--labels-root",
            str(costs),
            "--features-root",
            str(features),
            "--lock",
            str(args.lock),
            "--output",
            str(training_table),
        ],
        environment,
    )
    if args.prepare_only:
        print(f"\nPrepared training table: {training_table}")
        return
    run(
        [
            python,
            "pipeline/05_train_and_evaluate/run_joint_holdout.py",
            "worker",
            "--plan",
            "configs/ablation_plan.json",
            "--training-table",
            str(training_table),
            "--output-root",
            str(training),
            "--num-workers",
            "1",
            "--worker-id",
            "0",
        ],
        environment,
    )
    run(
        [
            python,
            "pipeline/05_train_and_evaluate/run_joint_holdout.py",
            "merge",
            "--plan",
            "configs/ablation_plan.json",
            "--training-table",
            str(training_table),
            "--output-root",
            str(training),
        ],
        environment,
    )
    run(
        [
            python,
            "pipeline/06_build_result_tables/summarize_ablation.py",
            "--plan",
            "configs/ablation_plan.json",
            "--training-root",
            str(training),
            "--output",
            str(args.results_root / "summary.csv"),
        ],
        environment,
    )
    run(
        [
            python,
            "pipeline/06_build_result_tables/compute_within_setting_kendall.py",
            "--predictions",
            str(training / "final20/predictions.parquet"),
            "--output",
            str(args.results_root / "kendall.csv"),
        ],
        environment,
    )
    print(f"\nComplete results: {args.results_root}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Export the configured MTEB dataset and model matrix with pinned runtimes."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mteb-root", type=Path, required=True)
    parser.add_argument("--vector-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--standard-python", required=True)
    parser.add_argument("--compat-python", required=True)
    parser.add_argument("--llm2vec-python", required=True)
    parser.add_argument("--tasks", nargs="*")
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("mteb_config.json")
    )
    return parser.parse_args()


def selected(values: list[str], filters: list[str] | None) -> list[str]:
    """Return all configured values or the requested subset."""
    if not filters:
        return values
    unknown = sorted(set(filters) - set(values))
    if unknown:
        raise ValueError(f"unknown selections: {unknown}")
    return [value for value in values if value in set(filters)]


def main() -> None:
    """Run one strict vector-export process for every selected pair."""
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    configured_tasks = [*config["datasets"], *config.get("external_datasets", [])]
    tasks = selected(config["datasets"] if not args.tasks else configured_tasks, args.tasks)
    model_names = [model["name"] for model in config["models"]]
    models = selected(model_names, args.models)
    runtime_python = {
        "standard": args.standard_python,
        "compat": args.compat_python,
        "llm2vec": args.llm2vec_python,
    }
    model_by_name = {model["name"]: model for model in config["models"]}
    exporter = Path(__file__).with_name("export_mteb_vectors.py")
    for task in tasks:
        for model_name in models:
            runtime = model_by_name[model_name]["runtime"]
            command = [
                runtime_python[runtime],
                str(exporter),
                "--mteb-root",
                str(args.mteb_root),
                "--vector-root",
                str(args.vector_root),
                "--cache-root",
                str(args.cache_root),
                "--config",
                str(args.config),
                "--task",
                task,
                "--model",
                model_name,
                "--device",
                args.device,
            ]
            print("\n$ " + " ".join(command), flush=True)
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

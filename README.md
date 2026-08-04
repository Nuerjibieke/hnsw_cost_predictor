# HNSW Query Cost Prediction

This project predicts per-query HNSW distance computations from index settings,
corpus size, and 16 vector-space features.

## Install

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Tested on x86-64 Linux with Python 3.10. Install `libgomp1` if LightGBM cannot
find `libgomp.so.1`.

## Run the small example

```bash
python run_pipeline.py
```

This runs four dataset/model pairs and all 112 HNSW settings. Outputs:

- `example/results/summary.csv`
- `example/results/kendall.csv`

The example results differ from the full paper results. Each run clears its
previous generated files.

## Code order

```text
vectors/                         text -> query and document vectors
pipeline/00_prepare_data/        full vectors -> optional compact example
pipeline/01_build_indexes/       vectors -> HNSW indexes
pipeline/02_measure_query_costs/ indexes -> distance counts
pipeline/03_extract_query_features/ vectors -> 16 query features
pipeline/04_build_training_table/ labels + features -> training rows
pipeline/05_train_and_evaluate/  training rows -> held-out predictions
pipeline/06_build_result_tables/ predictions -> paper tables
```

Shared code is in `src/hnsw_cost/`; PGTuner QPP is in `baseline/`; additional
experiments are in `experiments/`; reported tables are in `results/`.

## Run the 77-pair experiment

Generate vectors as described in `vectors/README.md`, then run:

```bash
python run_pipeline.py \
  --vector-root /path/to/vectors \
  --lock configs/vector_manifests.json \
  --output-root artifacts/full_run \
  --results-root artifacts/full_run/results
```

The grid is defined in `configs/hnsw_grid.json`. FAISS uses one thread, and
`hnsw_stats.ndis` is the target. Each test fold excludes its dataset and model
from training. Missing values use training-only medians. Targets, names, IDs,
recall, graph data, and search traces are not model inputs.

Feature formulas and model inputs are listed in `docs/FEATURE_DEFINITIONS.md`.

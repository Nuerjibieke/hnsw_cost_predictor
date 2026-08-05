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

This runs the complete pipeline on the compact vectors included in the
repository. Outputs:

- `example/results/summary.csv`
- `example/results/kendall.csv`

The example is a code check, not a paper result. Each run clears its previous
generated files.

## Code order

```text
vectors/                         MTEB vector-generation instructions
pipeline/00_prepare_data/        exported vectors -> optional compact example
pipeline/01_build_indexes/       vectors -> HNSW indexes
pipeline/02_measure_query_costs/ indexes -> distance counts
pipeline/03_extract_query_features/ vectors -> 16 query features
pipeline/04_build_training_table/ labels + features -> training rows
pipeline/05_train_and_evaluate/  training rows -> held-out predictions
pipeline/06_build_result_tables/ predictions -> paper tables
```

Shared code is in `src/hnsw_cost/`; PGTuner QPP is in `baseline/`; reported
tables are in `results/`.

## Run with exported vectors

Generate vectors as described in `vectors/README.md`. To inspect one dataset
without training, use a lock containing that dataset and run:

```bash
python run_pipeline.py \
  --vector-root /path/to/vectors \
  --lock /path/to/vector_lock.json \
  --output-root artifacts/run \
  --results-root artifacts/run/results \
  --prepare-only
```

This builds indexes, measures query costs, extracts features, and creates the
training table. Paper-style joint holdout training requires multiple datasets
and models because each test fold excludes both its dataset and its model.

Remove `--prepare-only` when the lock contains enough independent datasets and
models for this split.

The grid is defined in `configs/hnsw_grid.json`. FAISS uses one thread, and
`hnsw_stats.ndis` is the target. Each test fold excludes its dataset and model
from training. Missing values use training-only medians. Targets, names, IDs,
recall, graph data, and search traces are not model inputs.

Feature formulas and model inputs are listed in `docs/FEATURE_DEFINITIONS.md`.

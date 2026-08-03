# HNSW Query Search-Cost Prediction

This project predicts the number of HNSW distance calculations for each query.
The model uses three HNSW settings, the document count, and 16 features computed
from query and document vectors.

## Install

The code was tested on x86-64 Linux with Python 3.10.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

LightGBM needs the Linux OpenMP runtime. On Ubuntu or Debian, install
`libgomp1` if `libgomp.so.1` is missing.

## Run the example

```bash
python run_pipeline.py
```

The example uses 512 documents and 32 queries from each of four dataset/model
pairs. It runs all 112 HNSW settings and writes:

- `example/results/summary.csv`
- `example/results/kendall.csv`

Intermediate files are written under `example/output/`. The example is smaller
than the paper experiment, so its accuracy values are different. The paper
tables are under `results/`. The included tables were produced on x86-64 Linux.
Different CPU or math-library versions may give small floating-point differences.
Each run removes intermediate files left by an earlier run under `--output-root`.

## Pipeline

```text
vectors
  -> build HNSW indexes
  -> measure distance calculations
  -> compute query features
  -> build the training table
  -> train and test
  -> write result tables
```

- `pipeline/`: code for these six steps.
- `src/hnsw_cost/`: shared feature and model code.
- `baseline/`: PGTuner QPP baseline.
- `configs/`: HNSW settings, feature lists, vector list, and seed.
- `docs/`: feature definitions.

## Run all 77 dataset/model pairs

```bash
python run_pipeline.py \
  --vector-root /path/to/full/vector/root \
  --lock configs/vector_manifests.json \
  --output-root artifacts/full_run \
  --results-root artifacts/full_run/results
```

The 112 settings are:

- `M`: 8, 16, 32, 48
- `efConstruction`: 40, 80, 160, 320
- `efSearch`: 8, 16, 32, 64, 128, 256, 512

FAISS index building and search use one thread. HNSW counters are reset before
every query. The target is `faiss.cvar.hnsw_stats.ndis`.

For each test pair, training excludes rows with the same dataset and rows with
the same embedding model. Missing values are filled with medians from training
rows only. Search cost, dataset name, model name, and query ID are not inputs.

See `docs/FEATURE_DEFINITIONS.md` for the model inputs.

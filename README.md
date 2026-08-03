# HNSW Query-Level Search-Cost Prediction

This repository contains the executable pipeline for the paper's main HNSW
predictor, the PGTuner QPP baseline, and the reported result tables. The
predictor uses 20 inputs: three HNSW settings, corpus size, and 16 query-local
geometric features.

## Installation

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

On macOS, install the OpenMP runtime required by LightGBM first:

```bash
brew install libomp
```

## Project structure

- `run_pipeline.py`: run the complete pipeline with one command.
- `example/`: compact real-vector example and its final result tables.
- `baseline/`: PGTuner QPP feature calculation and evaluation code.
- `pipeline/00_prepare_data/`: create the compact example from locked vector exports.
- `pipeline/01_build_indexes/`: build the 16 HNSW indexes for each pair.
- `pipeline/02_measure_query_costs/`: collect per-query distance-computation labels.
- `pipeline/03_extract_query_features/`: compute the 16 query-local features.
- `pipeline/04_build_training_table/`: join labels and features into model input.
- `pipeline/05_train_and_evaluate/`: train models and evaluate held-out data.
- `pipeline/06_build_result_tables/`: summarize predictions into paper tables.
- `src/hnsw_cost/`: feature formulas and LightGBM model utilities.
- `configs/`: the HNSW grid, feature sets, vector lock, and random seed.
- `results/`: only the result tables and figure data reported in the paper.
- `docs/`: feature definitions and main-pipeline reproduction commands.

## Main pipeline

```text
embedding vectors
  -> 01_build_indexes
  -> 02_measure_query_costs
  -> 03_extract_query_features
  -> 04_build_training_table
  -> 05_train_and_evaluate
  -> 06_build_result_tables
```

The PGTuner QPP baseline is run separately; see `baseline/README.md`.

## Runnable example

The included example uses the first 512 documents and 32 queries from the
locked NFCorpus and SciFact exports for two embedding models. It retains the
paper's complete 112-setting HNSW grid, 16 query features, `log(1 + ndis)`
target, LightGBM configuration, and strict joint dataset/model holdout.

```bash
python run_pipeline.py
```

The two final tables are written to `example/results/summary.csv` and
`example/results/kendall.csv`. Intermediate indexes, measurements, the joined
training table, and fold predictions are written below `example/output/`.

The example demonstrates that every stage is executable on real paper vectors.
Its values are not the paper's reported accuracy values because it uses only a
small prefix of each vector collection. The reported full-workload tables remain
under `results/`. The included example tables were generated on x86-64 Linux;
small numerical differences can occur on other processor architectures.

To rebuild the compact input from the locked full exports:

```bash
python pipeline/00_prepare_data/prepare_example_data.py \
  --source-root /path/to/full/vector/root
```

To run the same pipeline on all 77 locked pairs:

```bash
python run_pipeline.py \
  --vector-root /path/to/full/vector/root \
  --lock configs/vector_manifests.json \
  --output-root artifacts/full_run \
  --results-root artifacts/full_run/results
```

The HNSW grid is
`M={8,16,32,48}` × `efConstruction={40,80,160,320}` ×
`efSearch={8,16,32,64,128,256,512}`. Index construction and search use one
FAISS OpenMP thread. Each query is searched separately; the HNSW statistics are
reset immediately before that query, and `faiss.cvar.hnsw_stats.ndis` is the
prediction target.

For a held dataset `d` and held embedding model `m`, training rows satisfy
`dataset != d AND model_slug != m`; the test rows are exactly `(d, m)`. Feature
imputation is fitted on the training partition only. The measured target and
dataset/model/query identifiers are never used as predictor inputs.

See `docs/REPRODUCIBILITY.md` for commands and
`docs/FEATURE_DEFINITIONS.md` for the feature formulas.

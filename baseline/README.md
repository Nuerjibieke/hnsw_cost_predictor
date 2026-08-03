# PGTuner QPP baseline

This folder contains the feature calculation and evaluation code for the
PGTuner QPP baseline in the paper. It is separate from the query-level method
under `pipeline/`.

The baseline uses 14 inputs: three HNSW settings, corpus and query counts,
embedding dimension, global LID, three dataset-structure statistics, and four
query-difficulty statistics. These are workload-level features, so the model
predicts one mean search cost for each HNSW configuration. That prediction is
assigned to every query in the configuration only when computing the paper's
query-level metrics.

Install PyTorch in the same environment as the main project:

```bash
pip install -r baseline/requirements.txt
```

After running the main pipeline to create `example/output/training_table.parquet`,
compute the baseline features and run its strict joint holdout evaluation:

```bash
python baseline/compute_features.py
python baseline/run.py
```

The example writes only `pair_features.csv`, `fold_metrics.csv`, and
`summary.csv` below `example/output/baseline/`.

For the complete 77-pair experiment:

```bash
python baseline/compute_features.py \
  --vector-root /path/to/full/vector/root \
  --lock configs/vector_manifests.json \
  --output artifacts/baseline/pair_features.csv \
  --device cuda

python baseline/run.py \
  --training-table artifacts/full_run/training_table.parquet \
  --pair-features artifacts/baseline/pair_features.csv \
  --output-root artifacts/baseline
```

Feature distances are exact Euclidean nearest-neighbor distances on the raw
stored embeddings, matching the baseline implementation. Each fold trains only
on rows whose dataset and embedding model both differ from the held pair.
Data-dependent normalization is fitted from that training fold only.

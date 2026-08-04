# QuoraRetrieval

Generate the seven Quora vector sets with `vectors/README.md`, then prepare the
labels and features:

```bash
python run_pipeline.py \
  --vector-root /path/to/quora_vectors \
  --lock configs/quora_vectors.json \
  --output-root artifacts/quora \
  --results-root artifacts/quora/results \
  --prepare-only
```

Train on the 11 main datasets and test each Quora model:

```bash
python experiments/quora/run.py \
  --training-table artifacts/full_run/training_table.parquet \
  --quora-table artifacts/quora/training_table.parquet \
  --output artifacts/quora/ours.csv
```

Build the baseline features and combine the three result rows:

```bash
python baseline/compute_features.py \
  --vector-root /path/to/quora_vectors \
  --lock configs/quora_vectors.json \
  --output artifacts/quora/baseline_features.csv \
  --device cuda

python baseline/run_external.py \
  --training-table artifacts/full_run/training_table.parquet \
  --training-features artifacts/baseline/pair_features.csv \
  --external-table artifacts/quora/training_table.parquet \
  --external-features artifacts/quora/baseline_features.csv \
  --output-root artifacts/quora/baseline

python experiments/quora/combine_results.py \
  --baseline artifacts/quora/baseline/summary.csv \
  --ours artifacts/quora/ours.csv \
  --output artifacts/quora/summary.csv
```

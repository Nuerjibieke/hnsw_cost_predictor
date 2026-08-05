# PGTuner QPP baseline

PGTuner QPP predicts one average cost for each dataset, model, and HNSW setting
from 14 configuration and dataset-level inputs.

## Run the example

```bash
pip install -r baseline/requirements.txt
python run_pipeline.py
python baseline/compute_features.py
python baseline/run.py
```

Outputs are written under `example/output/baseline/`.

## Run with exported vectors

```bash
python baseline/compute_features.py \
  --vector-root /path/to/vectors \
  --lock /path/to/vector_lock.json \
  --output artifacts/baseline/pair_features.csv \
  --device cuda

python baseline/run.py \
  --training-table artifacts/run/training_table.parquet \
  --pair-features artifacts/baseline/pair_features.csv \
  --output-root artifacts/baseline

python pipeline/06_build_result_tables/build_prediction_summary.py \
  --ablation artifacts/run/results/summary.csv \
  --baseline artifacts/baseline/summary.csv \
  --output artifacts/run/results/prediction_summary.csv
```

Features use exact Euclidean distances. Each fold excludes its test dataset and
model; scaling statistics use training rows only.

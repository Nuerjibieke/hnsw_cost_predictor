# PGTuner QPP baseline

The baseline predicts one average HNSW search cost for each dataset, embedding
model, and HNSW setting. It uses 14 inputs: three HNSW settings, document and
query counts, vector dimension, global LID, three document-distance values, and
four query-distance ratios.

## Run the example

After installing the main project, run:

```bash
pip install -r baseline/requirements.txt
python run_pipeline.py
python baseline/compute_features.py
python baseline/run.py
```

Outputs are written under `example/output/baseline/`.

## Run all 77 pairs

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

Features use exact Euclidean distances. For each test pair, training excludes
rows with the same dataset and rows with the same embedding model. Scaling
statistics are computed from training rows only.

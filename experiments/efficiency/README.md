# Runtime comparison

```bash
python experiments/efficiency/run.py \
  --training-table artifacts/full_run/training_table.parquet \
  --vector-root /path/to/vectors \
  --index-root artifacts/full_run/indexes \
  --output-root artifacts/efficiency
```

Direct time covers 16 builds and 112 batch searches. Prediction time covers
features, input assembly, and inference. Loading and training are excluded;
each stage is measured once, and FAISS and LightGBM use one CPU thread.

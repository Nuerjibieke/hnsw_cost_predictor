# Embedding model selection

```bash
python experiments/model_selection/run.py worker \
  --training-table artifacts/full_run/training_table.parquet \
  --output-root artifacts/model_selection

python experiments/model_selection/run.py merge \
  --training-table artifacts/full_run/training_table.parquet \
  --output-root artifacts/model_selection
```

Each of the 231 folds excludes the test dataset and both candidate models. The
merge writes 25,872 decisions and the reported table.

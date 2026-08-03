# Results

This directory is a compact archive of only the tables and figure data reported
in the paper. `run_pipeline.py` regenerates the main ablation and Kendall
outputs, while `baseline/` regenerates the PGTuner QPP baseline. The remaining
CSV files preserve the reported outputs of the paper's additional experiments.

- `prediction/`: main comparison summary and query-ordering Kendall result.
- `quora/`: QuoraRetrieval comparison summary.
- `ablation/`: feature-category ablation summary.
- `efficiency/`: timing data and the reported speedup summary.
- `model_selection/`: prediction-guided embedding-model selection summary.
- `other_indexes/`: NSG and Vamana comparison summary.

Fold diagnostics, feature importance, worst cases, intermediate predictions,
and unreported metrics are not included.

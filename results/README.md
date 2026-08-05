# Paper results

- `prediction/`: prediction metrics and Kendall's tau.
- `quora/`: QuoraRetrieval results.
- `ablation/`: feature-group results.
- `efficiency/`: running times and speedups.
- `model_selection/`: embedding-model selection results.
- `other_indexes/`: NSG and Vamana results.

These CSV files are the reported paper outputs; the compact example does not
reproduce them. `run_pipeline.py` writes ablation and Kendall tables for the
supplied vector lock. `baseline/` contains the PGTuner QPP implementation.

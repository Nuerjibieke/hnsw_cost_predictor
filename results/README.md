# Paper results

This folder contains the CSV tables reported in the paper.

- `prediction/`: prediction metrics and Kendall's tau.
- `quora/`: QuoraRetrieval results.
- `ablation/`: feature-group results.
- `efficiency/`: running times and speedups.
- `model_selection/`: embedding-model selection results.
- `other_indexes/`: NSG and Vamana results.

`run_pipeline.py` creates the main ablation and Kendall tables. `baseline/`
runs the PGTuner QPP baseline. Other CSV files are saved paper outputs.

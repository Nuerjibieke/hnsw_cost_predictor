# Paper results

- `prediction/`: prediction metrics and Kendall's tau.
- `quora/`: QuoraRetrieval results.
- `ablation/`: feature-group results.
- `efficiency/`: running times and speedups.
- `model_selection/`: embedding-model selection results.
- `other_indexes/`: NSG and Vamana results.

`run_pipeline.py` generates the ablation and Kendall tables. `baseline/`
contains the PGTuner QPP implementation. The other folders contain additional
tables reported in the paper.

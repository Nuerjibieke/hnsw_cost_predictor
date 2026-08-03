# Final feature definitions

All query-local features use cosine distance `d(q,x)=1-q·x` after L2
normalization.  Original distances determine ordering; `epsilon=1e-8` is used
only inside logarithms, denominators, reciprocals, and doubled-radius
thresholds.  Exact distance ties are broken by corpus offset.  The final model
does not use graph topology, HNSW traces, measured recall, hops, or cost.

## Four global-context inputs

`hnsw_ef_construction`, `hnsw_M`, `ef_search`, and `num_docs`.

## Eleven distance-growth inputs

- `lid20`: MLE LID using ranks 1–19 and rank 20 as the boundary.
- `adapt_lid_p005`, `adapt_lid_p01`, `adapt_lid_p02`, `adapt_lid_p05`: MLE LID
  at `k_p=min(N,max(20,ceil(pN)))`.
- `adapt_lid_profile_mean`, `adapt_lid_profile_max`,
  `adapt_lid_profile_slope`: summaries over the ordered profile
  `p=(0.001,0.002,0.005,0.01,0.02,0.05)`.
- `adapt_local_expansion_ratio_2r_p002`,
  `adapt_local_expansion_ratio_2r_p01`: `C_q(2 d_(k_p))/k_p`.
- `adapt_local_expansion_profile_mean`: mean of that ratio across all six
  profile scales.

The expansion statistic is explicitly rank-normalized.  Under boundary ties,
the denominator remains the selected rank `k_p`; it is not the exact count
`C_q(d_(k_p))`.  It must therefore not be described as the classical exact
ball-growth ratio.

## Two distance-separation inputs

- `rc10`: corpus-wide mean distance divided by the 10th-nearest distance.
- `dsp10`: mean distance among the nearest 10, multiplied by the sum of inverse
  distances outside the nearest 10 and normalized by `N`.

## Three directional inputs

`qdir_erank_k10`, `qdir_erank_k20`, and `qdir_erank_k40` are entropy effective
ranks of the Gram matrix formed from normalized query-to-neighbor displacement
vectors.  They are bounded by 1 and K when the displacement energy is nonzero.

The query-local feature names and ordering are defined by `QUERY_FEATURES` in
`src/hnsw_cost/features.py`.  Each estimator uses the feature order listed for
its variant in `configs/ablation_plan.json`.

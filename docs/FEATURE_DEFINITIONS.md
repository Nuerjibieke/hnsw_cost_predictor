# Model inputs

The model uses 20 inputs: four HNSW/corpus values and 16 query features. Vectors
are L2-normalized, and cosine distance is `d(q,x)=1-q·x`. Original distances set
the neighbor order. `epsilon=1e-8` only prevents division by zero and
`log(0)`. Nearest neighbors use the same NumPy partial ordering as the paper
experiments.

The inputs do not include measured cost, recall, HNSW hops, graph edges, or
search traces.

## HNSW and corpus

- `hnsw_ef_construction`: index `efConstruction`.
- `hnsw_M`: index `M`.
- `ef_search`: query `efSearch`.
- `num_docs`: document count.

## Distance growth

- `lid20`: LID from the nearest 20 documents.
- `adapt_lid_p005`, `adapt_lid_p01`, `adapt_lid_p02`, `adapt_lid_p05`: LID at
  rank `min(N,max(20,ceil(pN)))`.
- `adapt_lid_profile_mean`, `adapt_lid_profile_max`,
  `adapt_lid_profile_slope`: mean, maximum, and slope of LID at
  `p={0.001,0.002,0.005,0.01,0.02,0.05}`.
- `adapt_local_expansion_ratio_2r_p002`,
  `adapt_local_expansion_ratio_2r_p01`: number of documents within twice the
  selected neighbor distance, divided by the selected rank.
- `adapt_local_expansion_profile_mean`: mean expansion ratio at the six `p`
  values.

The expansion denominator remains the selected rank when boundary distances
are tied.

## Distance separation

- `rc10`: mean distance to all documents divided by the 10th-nearest distance.
- `dsp10`: mean distance to the nearest 10, multiplied by the sum of inverse
  distances to all remaining documents, then divided by `num_docs`.

## Neighbor directions

- `qdir_erank_k10`
- `qdir_erank_k20`
- `qdir_erank_k40`

For each `k`, query-to-neighbor vectors are scaled to length one. The feature is
the entropy effective rank of their Gram matrix. It is between 1 and `k` when
the vectors have nonzero length.

Feature order is defined in `src/hnsw_cost/features.py`. Feature sets are listed
in `configs/ablation_plan.json`.

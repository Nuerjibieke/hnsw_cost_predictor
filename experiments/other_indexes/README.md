# NSG and Vamana

NSG uses FAISS 1.10.0 with the included native counter patch:

```bash
git clone https://github.com/facebookresearch/faiss.git ../faiss
git -C ../faiss checkout v1.10.0
git -C ../faiss apply "$PWD/experiments/other_indexes/faiss_nsg_stats.patch"
cmake -S ../faiss -B ../faiss/build -DFAISS_ENABLE_GPU=OFF -DBUILD_TESTING=OFF
cmake --build ../faiss/build -j --target swigfaiss
python -m pip install ../faiss/build/faiss/python
```

Collect the native per-query counts:

```bash
python experiments/other_indexes/nsg.py \
  --faiss-root /path/to/faiss \
  --vector-root /path/to/vectors \
  --output-root artifacts/nsg_labels
```

Vamana uses DiskANN commit `78256bbab4685e1774e78d331e081a153be26823` with
the included counter patch:

```bash
git clone https://github.com/microsoft/DiskANN.git ../DiskANN
git -C ../DiskANN checkout 78256bbab4685e1774e78d331e081a153be26823
git -C ../DiskANN apply "$PWD/experiments/other_indexes/per_query_search_stats.patch"
cmake -S ../DiskANN -B ../DiskANN/build -DCMAKE_BUILD_TYPE=Release
cmake --build ../DiskANN/build -j --target build_memory_index search_memory_index
```

Collect labels and evaluate both indexes:

```bash
python experiments/other_indexes/vamana.py \
  --diskann-root /path/to/DiskANN \
  --vector-root /path/to/vectors \
  --output-root artifacts/vamana_labels

python experiments/other_indexes/evaluate.py \
  --index NSG \
  --labels-root artifacts/nsg_labels \
  --training-table artifacts/full_run/training_table.parquet \
  --output-root artifacts/nsg_results

python experiments/other_indexes/evaluate.py \
  --index Vamana \
  --labels-root artifacts/vamana_labels \
  --training-table artifacts/full_run/training_table.parquet \
  --output-root artifacts/vamana_results

python experiments/other_indexes/combine_results.py \
  --nsg artifacts/nsg_results/summary.csv \
  --vamana artifacts/vamana_results/summary.csv \
  --output artifacts/other_indexes_summary.csv
```

Both evaluations use the 16 query features and the joint dataset/model holdout.
Training rows are weighted so that each embedding model has equal total weight.

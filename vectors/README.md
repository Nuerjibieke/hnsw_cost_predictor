# Generate vectors with MTEB

[MTEB](https://github.com/embeddings-benchmark/mteb) is an external benchmark
and is not included in this repository. Use its framework to load each
retrieval task and encode its test queries and documents with the model
revision listed in the vector lock used for the run.

For a small example, export the test queries and documents from one retrieval
dataset, such as NFCorpus, with one model revision. This is sufficient for data
preparation, index construction, cost measurement, and feature extraction.

The paper vectors were exported with MTEB 2.11.6 at commit
`24365e91abc4021ab5286db039d5ae449b1ce190`. The standard environment used
PyTorch 2.9.0, Sentence Transformers 5.1.1, Transformers 4.44.2, Datasets
4.8.4, and Accelerate 1.13.0. Qwen3-Embedding-8B, F2LLM-v2-8B, and
GeoEmbedding used Sentence Transformers 3.3.1 and Transformers 4.51.3.
LLM2Vec 0.2.3 used Sheared-LLaMA-1.3B at `a4b76938edbf571ea7d7d9904861cbdca08809b4`
and its MNTP adapter at `eb4ee4c1f922be3c5961d26eb954d0755aa9b77c`.

The downstream pipeline expects one NumPy array and one aligned JSON Lines file
for queries and documents. See `example/data/vectors/` for the directory and
manifest format. The pipeline verifies the manifest and file hashes before use.

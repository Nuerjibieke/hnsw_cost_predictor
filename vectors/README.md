# Generate MTEB vectors

MTEB loads text, applies prompts, and encodes vectors. Its source is not copied
into this project. Run from the project root:

```bash
git clone https://github.com/embeddings-benchmark/mteb.git ../mteb
git -C ../mteb checkout 24365e91abc4021ab5286db039d5ae449b1ce190

python3.10 -m venv .venv/mteb-standard
.venv/mteb-standard/bin/python -m pip install -r vectors/requirements-standard.txt

python3.10 -m venv .venv/mteb-compat
.venv/mteb-compat/bin/python -m pip install -r vectors/requirements-compat.txt
```

Test one dataset/model pair:

```bash
python vectors/run_mteb_matrix.py \
  --mteb-root ../mteb \
  --vector-root artifacts/vectors \
  --cache-root artifacts/mteb_cache \
  --standard-python .venv/mteb-standard/bin/python \
  --compat-python .venv/mteb-compat/bin/python \
  --llm2vec-python .venv/mteb-standard/bin/python \
  --tasks NFCorpus \
  --models intfloat/e5-mistral-7b-instruct
```

Remove `--tasks` and `--models` to run all 77 pairs in `mteb_config.json`.

After all pairs finish, create the pipeline lock:

```bash
python vectors/build_vector_lock.py \
  --vector-root artifacts/vectors \
  --output configs/vector_manifests.json
```

The scripts verify revisions, dimensions, pair counts, and file hashes. The
LLM2Vec backbone and both adapters are pinned by commit.

Generate QuoraRetrieval separately:

```bash
python vectors/run_mteb_matrix.py \
  --mteb-root ../mteb \
  --vector-root artifacts/quora_vectors \
  --cache-root artifacts/mteb_cache \
  --standard-python .venv/mteb-standard/bin/python \
  --compat-python .venv/mteb-compat/bin/python \
  --llm2vec-python .venv/mteb-standard/bin/python \
  --tasks QuoraRetrieval

python vectors/build_vector_lock.py \
  --vector-root artifacts/quora_vectors \
  --tasks QuoraRetrieval \
  --output configs/quora_vectors.json
```

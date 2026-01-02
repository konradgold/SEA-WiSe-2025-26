# SEA-WiSe-2025-26
Repo for the search engine assignements

We use [uv](https://docs.astral.sh/uv/getting-started/installation/).

Download documents: 
```bash
wget https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-docs.tsv.gz
```
- Attention: Large (22gb when expanded)

Install sea package into the venv:
```python
uv pip install -e .
```

## Build the disk-based index
This is the official indexing pipeline. It parses the TSV, writes intermediate blocks to `data/blocks/`,
then merges them into the on-disk inverted index files under `data/` (e.g. `posting_list.bin`).

Run ingestion + index build:
```python
uv run python -m sea.ingest.pipeline
```

Possible issue if conda env active: Source current environment (`source .venv/bin/active`)

Tokenizer configuration is controlled via `configs/base.yaml`.

Run search:
```python
uv run python -m sea.query.search
```

## Performance benchmarking

Performance of key functions is printed automatically. The base.yaml specifies the performance run. To track the performance of different setups for many documents/queries we can use:

```bash
uv run python -m sea.perf.runner
```

## Profile application
Run the following line to track script performance
```bash
uv run python -m cProfile -o profile.pstats -m sea.ingest.pipeline
```
With the next command, you are able to view the performance measurement:
```bash
uvx snakeviz profile.pstats
```

## Learning to Rank (LTR)
The LTR pipeline implements a retrieve-then-rerank architecture. BM25 is used to retrieve top-N candidates, which are then re-scored by a TensorFlow-Ranking model to improve precision.

### 1. Split queries
Create deterministic splits for MS MARCO queries:
```bash
uv run python -m sea.ltr.split_cli --qrels data/msmarco-doctrain-qrels.tsv --out-dir data/splits
```

### 2. Pre-compute features
Generate feature caches (`.npz`) to speed up training. This retrieves candidates and extracts features once for a given split:
```bash
uv run python -m sea.ltr.prepare_data --split-file data/splits/train_qids.txt --out data/train_cache.npz
```

### 3. Train the model
Train the reranker using the pre-computed features. Training is integrated with W&B for logging:
```bash
uv run python -m sea.ltr.train_tfr --train-cache data/train_cache.npz --val-cache data/val_cache.npz
```

### 4. Search with reranking
Run a query through the full pipeline (BM25 -> Feature Extraction -> Model Scoring):
```bash
uv run python -m sea.ltr.serve_cli --model-path artifacts/my_model/model.keras --query "apple pie recipe"
```

Configuration for LTR (features, model architecture, training params) is managed in the `LTR` section of `configs/base.yaml`.


# SEA-WiSe-2025-26

Search engine for MS MARCO documents with BM25/semantic retrieval and optional LTR reranking.

We use [uv](https://docs.astral.sh/uv/getting-started/installation/).

## Setup

Download documents:
```bash
wget https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-docs.tsv.gz
gunzip msmarco-docs.tsv.gz
mv msmarco-docs.tsv data/
```
Note: Large file (~22GB when expanded)

Install package:
```bash
uv pip install -e .
```

## Quick Start

Build the index and run search:
```bash
uv run python -m sea.ingest.pipeline
uv run python -m sea.query.search_simple
```

## Indexing Pipeline

Build the disk-based inverted index from TSV documents:
```bash
uv run python -m sea.ingest.pipeline
```

This parses documents, writes intermediate blocks to `data/blocks/`, then merges them into the final index files. Use `--force` to overwrite existing index files.

Configuration in `configs/base.yaml`:
- `INGESTION.NUM_DOCUMENTS`: number of documents to index (-1 for all)
- `INGESTION.BATCH_SIZE`: batch size for parallel processing
- `SEARCH.FIELDED.ACTIVE`: enable fielded indexing (title/body/url)

## Search

Interactive search CLI:
```bash
uv run python -m sea.query.search_simple
```

The search strategy is controlled by `configs/base.yaml`:

| Setting | Options | Description |
|---------|---------|-------------|
| `SEARCH.RETRIEVAL` | `"bm25"`, `"semantic"` | Base retrieval method |
| `SEARCH.RERANKER.ENABLED` | `true`, `false` | Add LTR reranking on top |
| `SEARCH.EXPAND_QUERIES` | `true`, `false` | SPLADE query expansion (BM25 only) |

### BM25 Search

Default retrieval using BM25 with fielded search:
```yaml
SEARCH:
  RETRIEVAL: "bm25"
  FIELDED:
    ACTIVE: true
    WEIGHTS:
      title: 2.0
      body: 1.0
      url: 1.5
```

### Semantic Search

Vector search using pre-computed embeddings. Requires the embedding service running.

1. Start the embedding service:
```bash
uv run python -m sea.semantic.service
```

2. Compute document embeddings (one-time):
```bash
uv run python -m sea.semantic.index_cli
```
Use `--num_docs N` to limit, `--force` to overwrite, `--resume` to continue from checkpoint.

3. Enable semantic search in config:
```yaml
SEARCH:
  RETRIEVAL: "semantic"
SEMANTIC:
  SERVICE_URL: "http://localhost:8001"
  MODEL_ID: "nomic-ai/nomic-embed-text-v1.5"
  DIM: 64
  DEVICE: "mps"  # or "cuda:0", "cpu"
```

### SPLADE Query Expansion

Expand queries with related terms using SPLADE (BM25 only):
```yaml
SEARCH:
  RETRIEVAL: "bm25"
  EXPAND_QUERIES: true
SPLADE:
  MODEL_ID: "naver/splade-cocondenser-ensembledistil"
  DEVICE: "cuda:0"
```

## Learning to Rank (LTR)

The LTR pipeline implements a retrieve-then-rerank architecture. BM25 retrieves top-N candidates, which are re-scored by a TensorFlow-Ranking model.

### 1. Split queries

Create deterministic train/val/test splits:
```bash
uv run python -m sea.ltr.split_cli --qrels data/msmarco-doctrain-qrels.tsv --out-dir data/splits
```

### 2. Pre-compute features

Generate feature caches for training:
```bash
uv run python -m sea.ltr.prepare_data \
  --split-file data/splits/train_qids.txt \
  --out data/train_cache.npz

uv run python -m sea.ltr.prepare_data \
  --split-file data/splits/val_qids.txt \
  --out data/val_cache.npz
```
Use `--force` to overwrite existing cache files.

### 3. Train the model

Train the reranker (integrated with W&B for logging):
```bash
uv run python -m sea.ltr.train_tfr \
  --train-cache data/train_cache.npz \
  --val-cache data/val_cache.npz
```

### 4. Enable reranking

Add LTR reranking to search:
```yaml
SEARCH:
  RERANKER:
    ENABLED: true
    MODEL_PATH: "artifacts/tfr_reranker/model.keras"
    CANDIDATE_TOPN: 200
```

Or run a single query through the full pipeline:
```bash
uv run python -m sea.ltr.serve_cli --model-path artifacts/model.keras --query "apple pie recipe"
```

### LTR Features

The model uses 10 features per query-document pair:
- `bm25_score`: BM25 retrieval score
- `query_len`, `query_uniq_len`: query statistics
- `title_len`, `body_len`: document field lengths
- `title_overlap_cnt`, `body_overlap_cnt`: term overlap counts
- `body_overlap_ratio`: overlap as fraction of query
- `idf_body_overlap_sum`, `idf_title_overlap_sum`: IDF-weighted overlaps

## Configuration

All configuration is managed in `configs/base.yaml`. Key sections:

| Section | Description |
|---------|-------------|
| `TOKENIZER` | Text processing (lowercase, stopwords, stemming) |
| `SEARCH` | Retrieval method, result count, fielded weights |
| `SEMANTIC` | Embedding model, dimensions, service URL |
| `BM25` | K1 and B parameters |
| `LTR` | Training params, model architecture, features |
| `SPLADE` | Query expansion model and threshold |

## Profiling

Track script performance:
```bash
uv run python -m cProfile -o profile.pstats -m sea.ingest.pipeline
uvx snakeviz profile.pstats
```

## Testing

Run tests:
```bash
uv run pytest src/tests/ -v
```

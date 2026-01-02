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


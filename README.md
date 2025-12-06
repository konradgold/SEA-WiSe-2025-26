# SEA-WiSe-2025-26
Repo for the search engine assignements

We use [uv](https://docs.astral.sh/uv/getting-started/installation/).

Get the redis container: 
```bash
docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
```
- UI on localhost:8001, connect on localhost:6379

Connect to production:
```python
import redis
import dotenv
import os

dotenv.load_dotenv()

r = redis.Redis(
    host=os.getenv("REDIS_HOST_PROD", 
            'seenar.cloud.sci.hpi.de'), 
    port=os.getenv("REDIS_PORT", 6379), 
    password=os.getenv("REDIS_PW", None))

r.ping()
```
Requires corresponding env-variables.


Download documents: 
```bash
wget https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-docs.tsv.gz
```
- Attention: Large (22gb when expanded)

Recommended: Use .env file for redis port/path to documents

Install sea package into the venv:
```python
uv pip install -e .
```
Ingest documents naively using:
```python
uv run python -m sea.ingest.pipeline
```

Possible issue if conda env active: Source current environment (`source .venv/bin/active`)

Create the inverted index:
```python
# Run with --no-store-positions to only store term frequencies
uv run python -m sea.index.tokenizer_job
```
- Tokenizer configuration is controlled via the base.yaml variables:
  - `TOKENIZER_BACKEND` = `simple` (default) or `spacy`
  - `SPACY_MODEL` = name of spaCy model or `blank`
  - The tokenizer scans documents with keys matching `D*` and writes postings under `token:{token}`.
- Ingesting and tokenizing 10000 documents is <110MB

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


# SEA-WiSe-2025-26
Repo for the search engine assignements

We use [uv](https://docs.astral.sh/uv/getting-started/installation/).

Get the redis container: 
```bash
docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
```
- UI on localhost:8001, connect on localhost:6379

Download documents: 
```bash
wget https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-docs.tsv.gz
```
- Attention: Large (22gb when expanded)

Recommended: Use .env file for redis port/path to documents

Ingest documents naively using:
```python
uv run ingestion.py [--batch-size 500 --documents-path "..." --redis_port xyz]
```
Create the inverted index:
```python
uv run tokenize_redis_content.py
```
- if you want to use an AutoTokenizer from the transformer library, you can simply set that in your .env, e.g. TOKENIZER_MODEL=bert-base-cased.
- Ingesting and tokenizing 10000 documents is <110MB

Run search:
```python
uv run main.py
```

## Performance benchmarking

Performance of key functions is printed automatically. To track the performance of different setups for many documents/queries we can use:

```bash
# Ingest 1000 docs and return avg docs/min
python -m perf.runner --mode ingest --batch-size 1000 --documents-path msmarco-docs.tsv

# Query the index, 200 iterations cycling through queries in queries/sample_queries.txt, returns queries/min
python -m perf.runner --mode query --queries-path queries/sample_queries.txt --iterations 200 
```
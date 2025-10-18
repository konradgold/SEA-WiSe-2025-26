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
uv run main.py [--batch_size 500 --documents-path "..." --redis_port xyz]
```
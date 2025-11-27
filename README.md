# SEA-WiSe-2025-26
Repo for the search engine assignments

We use [uv](https://docs.astral.sh/uv/getting-started/installation/).

## Quick Start

### 1. Download MS MARCO Documents
```bash
wget https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-docs.tsv.gz
gunzip msmarco-docs.tsv.gz
```
- **Note**: ~22GB when expanded (~3.2M documents)

### 2. Install the Package
```bash
uv pip install -e .
```

### 3. Build the Disk Index
```bash
uv run python -m sea.index.build_index
```

This creates the following files in `index/`:
- `dictionary.bin` - Term -> posting location mapping (loaded into memory at startup)
- `postings.bin` - Posting lists (read from disk per query)
- `documents.bin` - Full document content (URL, title, body)
- `doc_offsets.bin` - Document ID -> file offset mapping

### 4. Run Search
```bash
uv run python -m sea.query.search
```

---

## Configuration

All settings are in `configs/base.yaml`:

```yaml
# Storage type: use "disk" for disk-based index
STORAGE:
  TYPE: "disk"  # options: "redis", "filesystem", "disk"

# Disk index settings
DISK_INDEX:
  PATH: "index/"           # Where index files are stored
  MEMORY_LIMIT_MB: 900     # Max RAM during indexing
  DOC_LIMIT: null          # Limit docs for testing (null = all)
```

### Testing with a Subset
To test with fewer documents, set `DOC_LIMIT` in the config:
```yaml
DISK_INDEX:
  DOC_LIMIT: 10000  # Only index first 10k documents
```

---

## Index File Formats

### Dictionary (`dictionary.bin`)
```
[num_terms: 4 bytes]
[term_len: 2B][term: variable][offset: 8B][length: 4B]
...
```
- Loaded fully into memory at startup (~50-100MB)
- Provides O(1) lookup from term to posting location

### Postings (`postings.bin`)
```
Per term:
[num_docs: 4B]
[doc_id: 4B][tf: 2B][num_pos: 2B][positions: 4B each]...
```
- Read from disk only when queried
- Sorted by doc_id for efficient intersection

### Documents (`documents.bin` + `doc_offsets.bin`)
```
[record_len: 4B][doc_id: 4B][url_len: 2B][url][title_len: 2B][title][body]
```
- Random access via offset table
- Only fetched for top-k results

---

## Alternative: In-Memory Index (for small datasets)

For development or small datasets, you can still use the in-memory filesystem storage:

```yaml
STORAGE:
  TYPE: "filesystem"
```

Then use the old pipeline:
```bash
# Ingest documents
uv run python -m sea.ingest.pipeline

# Build inverted index
uv run python -m sea.index.tokenizer_job

# Search
uv run python -m sea.query.search
```

---

## Redis Backend (Optional)

Get the redis container: 
```bash
docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
```
- UI on localhost:8001, connect on localhost:6379

Set in config:
```yaml
STORAGE:
  TYPE: "redis"
```

Connect to production:
```python
import redis
import dotenv
import os

dotenv.load_dotenv()

r = redis.Redis(
    host=os.getenv("REDIS_HOST_PROD", 'seenar.cloud.sci.hpi.de'), 
    port=os.getenv("REDIS_PORT", 6379), 
    password=os.getenv("REDIS_PW", None))

r.ping()
```
Requires corresponding env-variables.

---

## Performance Benchmarking

Performance of key functions is printed automatically. The base.yaml specifies the performance run. To track the performance of different setups:

```bash
uv run python -m sea.perf.runner
```

---

## Code Review Metrics

After running the indexer, it will report:
- **Index size**: Total size of index files
- **Indexing time**: Time to build the complete index
- **Startup time**: Time to load dictionary (shown at search startup)

Query timing is shown after each search.

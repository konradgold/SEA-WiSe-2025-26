"""
Build Disk Index from MS MARCO Documents

This is the main indexing script that:
1. Reads documents from the TSV file
2. Tokenizes each document 
3. Stores documents in the DocumentStore
4. Builds inverted index using SPIMI

Usage:
    python -m sea.index.build_index

The script respects memory limits and can handle the full MS MARCO dataset.
"""

import os
import sys
import time
import logging
import multiprocessing as mp
from typing import Iterator

from sea.index.document_store import DocumentStore
from sea.index.spimi import SPIMIIndexer
from sea.index.tokenization import get_tokenizer
from sea.utils.config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Global tokenizer for worker processes
_WORKER_TOKENIZER = None


def _init_tokenizer_worker():
    """Initialize tokenizer in worker process."""
    global _WORKER_TOKENIZER
    _WORKER_TOKENIZER = get_tokenizer()


def _tokenize_batch(batch: list[tuple]) -> list[tuple]:
    """
    Tokenize a batch of documents in a worker process.
    
    Input: [(doc_id, text), ...]
    Output: [(doc_id, tokens), ...]
    """
    global _WORKER_TOKENIZER
    if _WORKER_TOKENIZER is None:
        _WORKER_TOKENIZER = get_tokenizer()
    
    results = []
    for doc_id, text in batch:
        tokens = _WORKER_TOKENIZER.tokenize(text)
        results.append((doc_id, tokens))
    return results


def read_msmarco_documents(tsv_path: str, limit: int = None) -> Iterator[tuple]:
    """
    Read documents from MS MARCO TSV file.
    
    Yields (doc_id, url, title, body) tuples.
    """
    count = 0
    with open(tsv_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 4:
                continue
            
            doc_id, url, title, body = parts
            yield doc_id, url, title, body
            
            count += 1
            if limit is not None and count >= limit:
                break


def build_index(cfg) -> dict:
    """
    Build the disk-based index with parallel tokenization.
    """
    index_dir = cfg.DISK_INDEX.PATH
    tsv_path = cfg.DOCUMENTS
    memory_limit_mb = cfg.DISK_INDEX.MEMORY_LIMIT_MB
    doc_limit = cfg.DISK_INDEX.DOC_LIMIT
    num_workers = cfg.DISK_INDEX.NUM_WORKERS if cfg.DISK_INDEX.NUM_WORKERS else (mp.cpu_count() or 4)
    batch_size = cfg.DISK_INDEX.BATCH_SIZE if cfg.DISK_INDEX.BATCH_SIZE else 1000
    
    logger.info(f"Building index in: {index_dir}")
    logger.info(f"Reading from: {tsv_path}")
    logger.info(f"Memory limit: {memory_limit_mb} MB")
    logger.info(f"Using {num_workers} worker processes, batch size {batch_size}")
    if doc_limit:
        logger.info(f"Document limit: {doc_limit}")
    
    # Initialize components
    os.makedirs(index_dir, exist_ok=True)
    
    doc_store = DocumentStore(index_dir)
    doc_store.open_for_writing()
    
    indexer = SPIMIIndexer(index_dir, memory_limit_mb=memory_limit_mb)
    
    # Track statistics
    start_time = time.time()
    docs_processed = 0
    tokens_total = 0
    last_log_time = start_time
    
    # Create process pool for tokenization
    pool = mp.Pool(processes=num_workers, initializer=_init_tokenizer_worker)
    
    try:
        # Batch documents for parallel processing
        doc_batch = []  # [(doc_id, url, title, body), ...]
        tokenize_batch = []  # [(doc_id, text), ...]
        
        for doc_id, url, title, body in read_msmarco_documents(tsv_path, limit=doc_limit):
            doc_batch.append((doc_id, url, title, body))
            tokenize_batch.append((doc_id, f"{title} {body}"))
            
            if len(doc_batch) >= batch_size:
                # Process batch
                docs_processed, tokens_total = _process_batch(
                    pool, doc_batch, tokenize_batch,
                    doc_store, indexer, 
                    docs_processed, tokens_total
                )
                doc_batch = []
                tokenize_batch = []
                
                # Progress logging
                now = time.time()
                if now - last_log_time >= 10:
                    elapsed = now - start_time
                    rate = docs_processed / elapsed
                    logger.info(f"Progress: {docs_processed:,} docs, {rate:.0f} docs/sec, "
                               f"{tokens_total:,} tokens")
                    last_log_time = now
        
        # Process remaining batch
        if doc_batch:
            docs_processed, tokens_total = _process_batch(
                pool, doc_batch, tokenize_batch,
                doc_store, indexer,
                docs_processed, tokens_total
            )
    
    finally:
        pool.close()
        pool.join()
    
    # Finalize
    logger.info("Finalizing document store...")
    doc_store.finish_writing()
    
    logger.info("Finalizing index (merging blocks)...")
    indexer.finalize()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calculate index size
    index_size = 0
    for fname in ["dictionary.bin", "postings.bin", "documents.bin", "doc_offsets.bin"]:
        fpath = os.path.join(index_dir, fname)
        if os.path.exists(fpath):
            index_size += os.path.getsize(fpath)
    
    stats = {
        "documents_indexed": docs_processed,
        "tokens_total": tokens_total,
        "indexing_time_seconds": total_time,
        "indexing_time_formatted": format_time(total_time),
        "index_size_bytes": index_size,
        "index_size_formatted": format_size(index_size),
    }
    
    logger.info("=" * 60)
    logger.info("INDEXING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Documents indexed:  {stats['documents_indexed']:,}")
    logger.info(f"Total tokens:       {stats['tokens_total']:,}")
    logger.info(f"Indexing time:      {stats['indexing_time_formatted']}")
    logger.info(f"Index size:         {stats['index_size_formatted']}")
    logger.info("=" * 60)
    
    return stats


def _process_batch(pool, doc_batch, tokenize_batch, doc_store, indexer, docs_processed, tokens_total):
    """Process a batch of documents: store and tokenize in parallel."""
    # Store documents
    for doc_id, url, title, body in doc_batch:
        doc_store.add_document(doc_id, url, title, body)
    
    # Tokenize in parallel, split into chunks for workers
    num_workers = pool._processes
    chunk_size = max(1, len(tokenize_batch) // num_workers)
    chunks = [tokenize_batch[i:i + chunk_size] for i in range(0, len(tokenize_batch), chunk_size)]
    
    # Map to workers
    results = pool.map(_tokenize_batch, chunks)
    
    # Flatten and add to index
    for chunk_result in results:
        for doc_id, tokens in chunk_result:
            indexer.add_document(doc_id, tokens)
            tokens_total += len(tokens)
    
    docs_processed += len(doc_batch)
    return docs_processed, tokens_total


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time."""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.2f} hours"


def format_size(bytes_: int) -> str:
    """Format bytes into human-readable size."""
    if bytes_ < 1024:
        return f"{bytes_} B"
    elif bytes_ < 1024 * 1024:
        return f"{bytes_ / 1024:.1f} KB"
    elif bytes_ < 1024 * 1024 * 1024:
        return f"{bytes_ / 1024 / 1024:.1f} MB"
    else:
        return f"{bytes_ / 1024 / 1024 / 1024:.2f} GB"


def main():
    """Main entry point for building the index."""
    mp.freeze_support()  # For Windows compatibility
    
    cfg = Config(load=True)
    
    if not hasattr(cfg, "DISK_INDEX") or cfg.DISK_INDEX is None:
        logger.error("DISK_INDEX configuration not found. Please add it to your config file.")
        sys.exit(1)
    
    try:
        stats = build_index(cfg)
        return stats
    except KeyboardInterrupt:
        logger.warning("Indexing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        raise


if __name__ == "__main__":
    main()

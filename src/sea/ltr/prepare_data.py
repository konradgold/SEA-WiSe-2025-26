"""
Pre-compute LTR features and save to .npz for fast training.

Supports multiprocessing for significant speedup on multi-core machines.
Use --workers to control parallelism (default: number of CPUs - 2).
"""
import argparse
import multiprocessing as mp
import os
import sys
import traceback
from pathlib import Path
import random

import numpy as np
import tqdm

from sea.ltr.bm25 import BM25Retriever
from sea.ltr.candidates import iter_qids, load_qrels_map, load_queries_map
from sea.ltr.features import FeatureExtractor
from sea.utils.config_wrapper import Config

# Worker-local state (initialized once per process)
_retriever = None
_fe = None
_init_error = None


def _init_worker():
    """Initialize retriever and feature extractor once per worker process.

    Uses num_threads=1 to avoid nested parallelism (processes already parallelize work).
    """
    global _retriever, _fe, _init_error
    _init_error = None
    try:
        cfg = Config(load=True)
        cfg.SEARCH.VERBOSE_OUTPUT = False
        # Use single thread per worker to avoid oversubscription (N processes × M threads)
        _retriever = BM25Retriever.from_config(cfg, num_threads=1)
        _fe = FeatureExtractor.from_config(cfg)
    except Exception as e:
        _init_error = f"Worker init failed: {e}\n{traceback.format_exc()}"
        _retriever = None
        _fe = None


def _process_query(args_tuple):
    """Process a single query in a worker process.

    Returns:
        tuple: (features, labels) on success
        dict: {"error": message} on error
        None: on skip (no results, no positives, etc.)
    """
    qid, query, positives_internal, candidate_topn, list_size, seed = args_tuple

    # Check if worker initialization failed
    if _init_error:
        return {"error": _init_error}
    if _retriever is None:
        return {"error": "Retriever not initialized"}

    try:
        id_results = _retriever.retrieve_ids(query, topn=candidate_topn)
        if not id_results:
            return None

        return _sample_list_for_query(
            qid, query, id_results, positives_internal,
            _retriever, _fe, list_size, seed
        )
    except Exception as e:
        return {"error": f"Query {qid} failed: {e}\n{traceback.format_exc()}"}


def _sample_list_for_query(
    qid: int,
    query: str,
    id_results: list[tuple[int, float]],
    positives_internal: set[int],
    retriever,
    fe,
    list_size: int,
    seed: int,
):
    """Sample one positive and (list_size-1) negatives, extract features."""
    if list_size < 2:
        return None

    # Find positives in the retrieved set
    pos_int_ids = [int_id for int_id, _ in id_results if int_id in positives_internal]
    if not pos_int_ids:
        return None

    rng = random.Random((seed * 1_000_003) ^ qid)
    pos_id = rng.choice(pos_int_ids)

    neg_int_ids = [int_id for int_id, _ in id_results if int_id not in positives_internal]
    if not neg_int_ids:
        return None

    # Sample from hard negatives (top of the ranking)
    hard_pool_topk = 50
    pool = neg_int_ids[:min(hard_pool_topk, len(neg_int_ids))]
    num_neg = list_size - 1

    if len(pool) >= num_neg:
        negs = rng.sample(pool, num_neg)
    else:
        negs = [rng.choice(pool) for _ in range(num_neg)]

    score_map = dict(id_results)
    chosen_pairs = [(pos_id, score_map[pos_id])] + [(nid, score_map[nid]) for nid in negs]
    rng.shuffle(chosen_pairs)

    # Hydrate documents from disk
    docs = retriever.hydrate_docs(chosen_pairs)
    if len(docs) != list_size:
        return None

    labels = np.array([1.0 if p[0] == pos_id else 0.0 for p in chosen_pairs], dtype=np.float32)
    features = fe.extract_many(query, docs).astype(np.float32, copy=False)

    return features, labels


def _run_extraction(work_items, num_workers):
    """Run feature extraction with single or multiple workers.

    Uses Pool.imap_unordered with chunking for efficient batched processing
    and smooth progress bar updates.
    """
    all_features = []
    all_labels = []
    skipped = 0
    errors = 0
    first_error = None

    def handle_result(result):
        nonlocal skipped, errors, first_error
        if result is None:
            skipped += 1
            return None
        if isinstance(result, dict) and "error" in result:
            errors += 1
            if first_error is None:
                first_error = result["error"]
            return None
        return result

    if num_workers == 1:
        print("Initializing worker (single-process mode)...")
        _init_worker()
        if _init_error:
            print(f"ERROR: {_init_error}")
            return [], [], 0
        print("Worker initialized, starting feature extraction...")

        for item in tqdm.tqdm(work_items, desc="Extracting features"):
            result = handle_result(_process_query(item))
            if result:
                all_features.append(result[0])
                all_labels.append(result[1])
    else:
        mp_ctx = mp.get_context("spawn" if sys.platform == "darwin" else "forkserver")
        # Use small chunksize for responsive progress updates (1 item per result)
        chunksize = 1

        print(f"Spawning {num_workers} worker processes (this may take a moment)...")

        with mp_ctx.Pool(processes=num_workers, initializer=_init_worker) as pool:
            results_iter = pool.imap_unordered(_process_query, work_items, chunksize=chunksize)
            pbar = tqdm.tqdm(results_iter, total=len(work_items), desc="Extracting features",
                             smoothing=0, miniters=1)
            for result in pbar:
                result = handle_result(result)
                if result:
                    all_features.append(result[0])
                    all_labels.append(result[1])

    if errors > 0:
        print(f"\nWARNING: {errors} queries failed with errors.")
        if first_error:
            print(f"First error:\n{first_error}")

    return all_features, all_labels, skipped


def main():
    cfg = Config(load=True)
    cfg.SEARCH.VERBOSE_OUTPUT = False

    ap = argparse.ArgumentParser(description="Pre-compute LTR features and save to .npz.")
    ap.add_argument(
        "--queries", type=str, default=cfg.LTR.QUERIES if hasattr(cfg, "LTR") else None, help="Path to queries TSV."
    )
    ap.add_argument(
        "--qrels", type=str, default=cfg.LTR.QRELS if hasattr(cfg, "LTR") else None, help="Path to qrels file."
    )
    ap.add_argument(
        "--split-file",
        type=str,
        default=None,
        help="Path to file with one qid per line (e.g., train_qids.txt).",
    )
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--list-size", type=int, default=cfg.LTR.LIST_SIZE if hasattr(cfg, "LTR") else 100)
    ap.add_argument("--candidate-topn", type=int, default=cfg.LTR.CANDIDATE_TOPN if hasattr(cfg, "LTR") else 200)
    ap.add_argument("--seed", type=int, default=cfg.LTR.SEED if hasattr(cfg, "LTR") else 42)
    ap.add_argument("--force", action="store_true", help="Overwrite existing output file")
    ap.add_argument("--workers", type=int, default=0, help="Number of worker processes (0 = auto, 1 = single-process)")
    args = ap.parse_args()

    if not (args.queries and args.qrels and args.split_file):
        ap.error("Must provide --queries, --qrels, and --split-file (or define in base.yaml).")

    out_path = Path(args.out)
    if out_path.exists() and not args.force:
        print(f"Error: Output file already exists: {args.out}")
        print("\nUse --force to overwrite existing file.")
        sys.exit(1)

    num_workers = args.workers if args.workers > 0 else max(1, (os.cpu_count() or 4) - 2)

    print("Loading queries and qrels...")
    queries = load_queries_map(args.queries)
    qrels = load_qrels_map(args.qrels)
    qids = list(iter_qids(args.split_file))

    if args.limit > 0:
        qids = qids[:args.limit]

    # Load retriever in main process to get doc metadata mapping
    print("Initializing retriever for qrels mapping...")
    retriever = BM25Retriever.from_config(cfg)

    # Convert qrels to internal IDs once (avoids repeated lookups)
    print("Mapping qrels to internal IDs...")
    storage_mgr = next(iter(retriever.ranker.storage_managers.values()))
    doc_metadata = storage_mgr.getDocMetadata()
    if not doc_metadata:
        print("Error: Document metadata is empty. Please re-run ingestion.")
        return
    orig_to_int = {meta[0]: doc_id for doc_id, meta in doc_metadata.items()}

    qrels_internal = {}
    unmapped_docs = 0
    for qid, orig_ids in qrels.items():
        internal_ids = {orig_to_int[oid] for oid in orig_ids if oid in orig_to_int}
        unmapped_docs += len(orig_ids) - len(internal_ids)
        if internal_ids:
            qrels_internal[qid] = internal_ids

    print(f"  Loaded {len(queries)} queries, {len(qrels)} qrels entries")
    print(f"  Mapped {len(qrels_internal)} qrels to internal IDs ({unmapped_docs} docs not in index)")
    print(f"  Document index has {len(orig_to_int)} documents")

    # Build work items
    work_items = [
        (qid, queries[qid], qrels_internal[qid], args.candidate_topn, args.list_size, args.seed)
        for qid in qids
        if qid in queries and qid in qrels_internal
    ]

    # Diagnose why work items might be missing
    if len(work_items) < len(qids):
        missing_queries = sum(1 for qid in qids if qid not in queries)
        missing_qrels = sum(1 for qid in qids if qid not in qrels_internal)
        print(f"  Note: {len(qids)} qids requested, {len(work_items)} have both query text and qrels")
        if missing_queries > 0:
            print(f"    - {missing_queries} qids missing from queries file")
        if missing_qrels > 0:
            print(f"    - {missing_qrels} qids missing from qrels (or all their docs not in index)")

    print(f"Processing {len(work_items)} queries with {num_workers} workers...")

    all_features, all_labels, skipped = _run_extraction(work_items, num_workers)

    if not all_features:
        print(f"Error: No samples generated from {len(work_items)} queries (skipped {skipped}).")
        return

    print(f"Generated {len(all_features)} samples (skipped {skipped} queries)")
    print(f"Saving to {args.out}...")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        features=np.array(all_features, dtype=np.float32),
        labels=np.array(all_labels, dtype=np.float32)
    )
    print(f"Successfully saved {len(all_features)} samples to {args.out}")


if __name__ == "__main__":
    main()

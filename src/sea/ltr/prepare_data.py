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
_feature_extractor = None
_init_error = None


def _init_worker():
    """Initialize retriever and feature extractor once per worker process.

    Uses num_threads=1 to avoid nested parallelism (processes already parallelize work).
    """
    global _retriever, _feature_extractor, _init_error
    _init_error = None
    try:
        config = Config(load=True)
        config.SEARCH.VERBOSE_OUTPUT = False
        # Use single thread per worker to avoid oversubscription (N processes × M threads)
        _retriever = BM25Retriever.from_config(config, num_threads=1)
        _feature_extractor = FeatureExtractor.from_config(config)
    except Exception as e:
        _init_error = f"Worker init failed: {e}\n{traceback.format_exc()}"
        _retriever = None
        _feature_extractor = None


def _process_query(args_tuple):
    """Process a single query in a worker process.

    Returns:
        tuple: (features, labels) on success
        dict: {"error": message} on error
        None: on skip (no results, no positives, etc.)
    """
    query_id, query_text, positives_internal, candidate_top_n, list_size, seed = args_tuple

    # Check if worker initialization failed
    if _init_error:
        return {"error": _init_error}
    if _retriever is None:
        return {"error": "Retriever not initialized"}

    try:
        id_score_results = _retriever.retrieve_ids(query_text, topn=candidate_top_n)
        if not id_score_results:
            return None

        return _sample_list_for_query(
            query_id, query_text, id_score_results, positives_internal,
            _retriever, _feature_extractor, list_size, seed
        )
    except Exception as e:
        return {"error": f"Query {query_id} failed: {e}\n{traceback.format_exc()}"}


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
    """Sample one positive and (list_size-1) hard negatives, then extract features.

    Hard negative sampling: pick negatives from the top-50 BM25 results that are
    irrelevant
    """
    if list_size < 2:
        return None

    # Separate retrieved docs into positives and negatives
    positive_ids_in_results = [internal_id for internal_id, _ in id_results if internal_id in positives_internal]
    if not positive_ids_in_results:
        return None

    negative_ids_in_results = [internal_id for internal_id, _ in id_results if internal_id not in positives_internal]
    if not negative_ids_in_results:
        return None

    # reproducable seed based on qid
    random_generator = random.Random((seed * 1_000_003) ^ qid)
    selected_positive_id = random_generator.choice(positive_ids_in_results)

    # Hard negative pool: top-50 non-relevant docs (highest BM25 scores among negatives)
    hard_negative_pool_size = 50
    hard_negative_pool = negative_ids_in_results[:hard_negative_pool_size]
    num_negatives_needed = list_size - 1

    if len(hard_negative_pool) >= num_negatives_needed:
        selected_negative_ids = random_generator.sample(hard_negative_pool, num_negatives_needed)
    else:
        # If pool is smaller than needed sample with replacement
        selected_negative_ids = [random_generator.choice(hard_negative_pool) for _ in range(num_negatives_needed)]

    id_to_score = dict(id_results)
    selected_id_score_pairs = [(selected_positive_id, id_to_score[selected_positive_id])]
    selected_id_score_pairs += [(negative_id, id_to_score[negative_id]) for negative_id in selected_negative_ids]
    random_generator.shuffle(selected_id_score_pairs)

    # Hydrate documents from disk
    documents = retriever.hydrate_docs(selected_id_score_pairs)
    if len(documents) != list_size:
        return None

    # 1.0 for the positive document, 0.0 for negatives
    labels = np.array([1.0 if pair[0] == selected_positive_id else 0.0 for pair in selected_id_score_pairs], dtype=np.float32)
    features = fe.extract_many(query, documents).astype(np.float32, copy=False)

    return features, labels


def _run_extraction(work_items, num_workers):
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
        print(f"Spawning {num_workers} worker processes (this may take a moment)...")

        with mp_ctx.Pool(processes=num_workers, initializer=_init_worker) as pool:
            results_iter = pool.imap_unordered(_process_query, work_items, chunksize=1)
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
    config = Config(load=True)
    config.SEARCH.VERBOSE_OUTPUT = False

    parser = argparse.ArgumentParser(description="Pre-compute LTR features and save to .npz.")
    parser.add_argument(
        "--queries", type=str, default=config.LTR.QUERIES if hasattr(config, "LTR") else None, help="Path to queries TSV."
    )
    parser.add_argument(
        "--qrels", type=str, default=config.LTR.QRELS if hasattr(config, "LTR") else None, help="Path to qrels file."
    )
    parser.add_argument(
        "--split-file",
        type=str,
        default=None,
        help="Path to file with one qid per line (e.g., train_qids.txt).",
    )
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--list-size", type=int, default=config.LTR.LIST_SIZE if hasattr(config, "LTR") else 100)
    parser.add_argument("--candidate-topn", type=int, default=config.LTR.CANDIDATE_TOPN if hasattr(config, "LTR") else 200)
    parser.add_argument("--seed", type=int, default=config.LTR.SEED if hasattr(config, "LTR") else 42)
    parser.add_argument("--force", action="store_true", help="Overwrite existing output file")
    parser.add_argument("--workers", type=int, default=0, help="Number of worker processes (0 = auto, 1 = single-process)")
    args = parser.parse_args()

    if not (args.queries and args.qrels and args.split_file):
        parser.error("Must provide --queries, --qrels, and --split-file (or define in base.yaml).")

    output_path = Path(args.out)
    if output_path.exists() and not args.force:
        print(f"Error: Output file already exists: {args.out}")
        print("\nUse --force to overwrite existing file.")
        sys.exit(1)

    num_workers = args.workers if args.workers > 0 else max(1, (os.cpu_count() or 4) - 2)

    print("Loading queries and qrels...")
    query_id_to_text = load_queries_map(args.queries)
    query_id_to_relevant_docs = load_qrels_map(args.qrels)
    query_ids = list(iter_qids(args.split_file))

    if args.limit > 0:
        query_ids = query_ids[:args.limit]

    # Load retriever in main process to get doc metadata mapping
    print("Initializing retriever for qrels mapping...")
    retriever = BM25Retriever.from_config(config)

    # Convert qrels to internal IDs once
    print("Mapping qrels to internal IDs...")
    storage_manager = next(iter(retriever.ranker.storage_managers.values()))
    document_metadata = storage_manager.getDocMetadata()
    if not document_metadata:
        print("Error: Document metadata is empty. Please re-run ingestion.")
        return
    original_id_to_internal_id = {meta[0]: doc_id for doc_id, meta in document_metadata.items()}

    qrels_with_internal_ids = {}
    unmapped_document_count = 0
    for query_id, original_doc_ids in query_id_to_relevant_docs.items():
        internal_ids = {original_id_to_internal_id[orig_id] for orig_id in original_doc_ids if orig_id in original_id_to_internal_id}
        unmapped_document_count += len(original_doc_ids) - len(internal_ids)
        if internal_ids:
            qrels_with_internal_ids[query_id] = internal_ids

    print(f"  Loaded {len(query_id_to_text)} queries, {len(query_id_to_relevant_docs)} qrels entries")
    print(f"  Mapped {len(qrels_with_internal_ids)} qrels to internal IDs ({unmapped_document_count} docs not in index)")
    print(f"  Document index has {len(original_id_to_internal_id)} documents")

    # Build work items (query_id, query_text, positive_ids, candidate_topn, list_size, seed)
    work_items = [
        (query_id, query_id_to_text[query_id], qrels_with_internal_ids[query_id], args.candidate_topn, args.list_size, args.seed)
        for query_id in query_ids
        if query_id in query_id_to_text and query_id in qrels_with_internal_ids
    ]

    # Diagnose why work items might be missing
    if len(work_items) < len(query_ids):
        missing_query_text = sum(1 for query_id in query_ids if query_id not in query_id_to_text)
        missing_relevance_labels = sum(1 for query_id in query_ids if query_id not in qrels_with_internal_ids)
        print(f"  Note: {len(query_ids)} qids requested, {len(work_items)} have both query text and qrels")
        if missing_query_text > 0:
            print(f"    - {missing_query_text} qids missing from queries file")
        if missing_relevance_labels > 0:
            print(f"    - {missing_relevance_labels} qids missing from qrels (or all their docs not in index)")

    print(f"Processing {len(work_items)} queries with {num_workers} workers...")

    all_features, all_labels, skipped_count = _run_extraction(work_items, num_workers)

    if not all_features:
        print(f"Error: No samples generated from {len(work_items)} queries (skipped {skipped_count}).")
        return

    print(f"Generated {len(all_features)} samples (skipped {skipped_count} queries)")
    print(f"Saving to {args.out}...")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        features=np.array(all_features, dtype=np.float32),
        labels=np.array(all_labels, dtype=np.float32)
    )
    print(f"Successfully saved {len(all_features)} samples to {args.out}")


if __name__ == "__main__":
    main()

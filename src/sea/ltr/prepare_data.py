"""
Pre-compute LTR features and save to .npz for fast training.

This script is optimized for single-process execution with efficient I/O batching.
Multiprocessing was causing disk contention during worker initialization.
"""
import argparse
import numpy as np
from pathlib import Path
import random
import sys
import tqdm

from sea.ltr.candidates import load_qrels_map, load_queries_map, iter_qids
from sea.ltr.bm25 import BM25Retriever
from sea.ltr.features import FeatureExtractor
from sea.utils.config_wrapper import Config


def sample_list_for_query(
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
    args = ap.parse_args()

    if not (args.queries and args.qrels and args.split_file):
        ap.error("Must provide --queries, --qrels, and --split-file (or define in base.yaml).")

    # Check if output file already exists
    out_path = Path(args.out)
    if out_path.exists() and not args.force:
        print(f"Error: Output file already exists: {args.out}")
        print("\nUse --force to overwrite existing file.")
        sys.exit(1)

    print("Loading queries and qrels...")
    queries = load_queries_map(args.queries)
    qrels = load_qrels_map(args.qrels)
    qids = list(iter_qids(args.split_file))

    if args.limit > 0:
        qids = qids[:args.limit]

    retriever = BM25Retriever.from_config(cfg)
    fe = FeatureExtractor.from_config(cfg)

    # Convert qrels to internal IDs once (avoids repeated lookups)
    print("Mapping qrels to internal IDs...")
    # Use the first available storage manager to get doc metadata
    storage_mgr = next(iter(retriever.ranker.storage_managers.values()))
    doc_metadata = storage_mgr.getDocMetadata()
    if not doc_metadata:
        print("Error: Document metadata is empty. Please re-run ingestion.")
        return
    orig_to_int = {meta[0]: doc_id for doc_id, meta in doc_metadata.items()}

    qrels_internal = {}
    for qid, orig_ids in qrels.items():
        internal_ids = {orig_to_int[oid] for oid in orig_ids if oid in orig_to_int}
        if internal_ids:
            qrels_internal[qid] = internal_ids

    all_features = []
    all_labels = []
    skipped = 0

    print(f"Processing {len(qids)} queries...")
    for qid in tqdm.tqdm(qids, desc="Extracting features"):
        query = queries.get(qid)
        positives_internal = qrels_internal.get(qid)

        if query is None or not positives_internal:
            skipped += 1
            continue

        # Retrieve candidate IDs (fast, no document content loaded)
        id_results = retriever.retrieve_ids(query, topn=args.candidate_topn)
        if not id_results:
            skipped += 1
            continue

        # Sample and extract features
        result = sample_list_for_query(
            qid=qid,
            query=query,
            id_results=id_results,
            positives_internal=positives_internal,
            retriever=retriever,
            fe=fe,
            list_size=args.list_size,
            seed=args.seed,
        )

        if result is None:
            skipped += 1
            continue

        features, labels = result
        all_features.append(features)
        all_labels.append(labels)

    if not all_features:
        print(f"Error: No samples generated from {len(qids)} queries (skipped {skipped}).")
        return

    print(f"Generated {len(all_features)} samples (skipped {skipped} queries)")
    print(f"Saving to {args.out}...")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        args.out,
        features=np.array(all_features, dtype=np.float32),
        labels=np.array(all_labels, dtype=np.float32)
    )
    print(f"Successfully saved {len(all_features)} samples to {args.out}")


if __name__ == "__main__":
    main()

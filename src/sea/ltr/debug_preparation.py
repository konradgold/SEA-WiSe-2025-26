import sys
from pathlib import Path
import hydra
import tqdm

# Ensure we can import from the current project
sys.path.append(str(Path(__file__).parent.parent.parent))

from sea.ltr.candidates import load_qrels_map, load_queries_map, iter_qids
from sea.ltr.bm25 import BM25Retriever
from sea.storage.IO import DocDictonaryIO
import hydra


@hydra.main(config_path="../../../configs", config_name="debug_preparation", version_base=None)
def main(cfg):
    cfg.SEARCH.VERBOSE_OUTPUT = False

    print(f"--- Configuration ---")
    print(f"Queries: {cfg.queries}")
    print(f"Qrels: {cfg.qrels}")
    print(f"Split File: {cfg.split_file}")
    print(f"Limit: {cfg.limit}")
    print(f"Target Top-N: {cfg.candidate_topn}")
    print("----------------------\n")

    print("Loading data...")
    queries = load_queries_map(cfg.queries)
    qrels = load_qrels_map(cfg.qrels)
    all_qids = list(iter_qids(cfg.split_file))
    qids = all_qids[:cfg.limit] if cfg.limit > 0 else all_qids

    print(f"Loaded {len(queries)} queries, {len(qrels)} qrels groups.")
    print(f"Analyzing {len(qids)} QIDs from split file...\n")

    print("Mapping qrels to internal IDs...")
    doc_io = DocDictonaryIO(rewrite=False, cfg=cfg)
    doc_metadata = doc_io.read()
    orig_to_int = {meta[0]: doc_id for doc_id, meta in doc_metadata.items()}
    doc_io.close()
    print(f"Index contains {len(orig_to_int)} documents.\n")

    stats = {
        "1_total": 0,
        "2_has_query": 0,
        "3_has_qrels": 0,
        "4_in_index": 0,
        "5_in_top_n": 0
    }

    retriever = BM25Retriever.from_config(cfg)
    
    # Check a few more thresholds for sensitivity
    thresholds = [10, 50, 100, cfg.candidate_topn, 500, 1000]
    thresholds = sorted(list(set(thresholds)))
    recall_counts = {t: 0 for t in thresholds}

    for qid in tqdm.tqdm(qids, desc="Analyzing Funnel"):
        stats["1_total"] += 1
        
        # 1. Has Query Text
        query_text = queries.get(qid)
        if not query_text:
            continue
        stats["2_has_query"] += 1
        
        # 2. Has Qrels
        pos_orig_ids = qrels.get(qid)
        if not pos_orig_ids:
            continue
        stats["3_has_qrels"] += 1
        
        # 3. Positives in Index
        pos_int_ids = {orig_to_int[oid] for oid in pos_orig_ids if oid in orig_to_int}
        if not pos_int_ids:
            continue
        stats["4_in_index"] += 1
        
        # 4. Recall Analysis
        max_t = max(thresholds)
        results = retriever.retrieve_ids(query_text, topn=max_t)
        retrieved_ids = [res[0] for res in results]
        
        for t in thresholds:
            top_t_ids = set(retrieved_ids[:t])
            if any(pid in top_t_ids for pid in pos_int_ids):
                recall_counts[t] += 1
        
        if any(pid in set(retrieved_ids[:cfg.candidate_topn]) for pid in pos_int_ids):
            stats["5_in_top_n"] += 1

    print("\n--- Funnel Analysis ---")
    print(f"{'Step':<25} | {'Count':<10} | {'Retention %':<12} | {'Step Loss %':<12}")
    print("-" * 68)
    
    prev_count = len(qids)
    steps = [
        ("Total QIDs analyzed", stats["1_total"]),
        ("Has Query Text", stats["2_has_query"]),
        ("Has Qrels", stats["3_has_qrels"]),
        ("Positives in Index", stats["4_in_index"]),
        (f"In Top-{cfg.candidate_topn} (Yield)", stats["5_in_top_n"])
    ]

    for name, count in steps:
        retention = (count / len(qids)) * 100
        loss = ((count / prev_count) * 100) if prev_count > 0 else 0
        print(f"{name:<25} | {count:<10} | {retention:>10.2f}% | {loss:>10.2f}%")
        prev_count = count

    print("\n--- Recall Sensitivity Analysis ---")
    print("Shows how yield would change if you adjusted --candidate-topn")
    print(f"{'Top-K':<10} | {'Queries':<10} | {'Recall@K %*':<12} | {'Total Yield %':<12}")
    print("-" * 55)
    for t in thresholds:
        recall_at_k = (recall_counts[t] / stats["4_in_index"] * 100) if stats["4_in_index"] > 0 else 0
        yield_pct = (recall_counts[t] / len(qids)) * 100
        print(f"{t:<10} | {recall_counts[t]:<10} | {recall_at_k:>10.2f}% | {yield_pct:>10.2f}%")
    
    print("\n* Recall@K % is relative to queries that have positives in the index.")
    
    print("\n--- Recommendations ---")
    if stats["2_has_query"] < stats["1_total"] * 0.9:
        print("!! Large drop in 'Has Query Text'. Check if your split file matches your queries.tsv.")
    if stats["3_has_qrels"] < stats["2_has_query"] * 0.9:
        print("!! Large drop in 'Has Qrels'. Check if your split file matches your qrels.tsv.")
    if stats["4_in_index"] < stats["3_has_qrels"] * 0.8:
        print("!! Large drop in 'Positives in Index'. You haven't indexed all documents needed for training.")
    
    yield_pct = (stats["5_in_top_n"] / len(qids)) * 100
    if yield_pct < 50:
        print(f"!! Low Yield ({yield_pct:.1f}%). Most common cause is the Recall Gap.")
        print(f"   Suggestion: Increase --candidate-topn or increase input --limit to compensate.")

if __name__ == "__main__":
    main()


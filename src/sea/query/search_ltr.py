import sys
import time
from pathlib import Path
from typing import Optional

import hydra
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from sea.index.tokenization import TokenizerAbstract, get_tokenizer
from sea.query.splade import SpladeEncoder
from sea.ranking.io_wrapper import BM25
from sea.ranking.ranking import BM25Ranking
from sea.utils.chunker import Chunker


def search_documents(
    *,
    retriever,
    reranker,
    query: str,
    splade_encoder: Optional[SpladeEncoder],
    tokenizer: TokenizerAbstract,
    use_reranker: bool,
    candidate_topn: int,
    final_topk: int
):
    """
    Search documents with optional SPLADE expansion and LTR reranking.
    
    Args:
        retriever: BM25 retriever function
        reranker: TFRReranker instance (can be None if not using reranking)
        query: User query string
        splade_encoder: SPLADE encoder for query expansion (can be None)
        tokenizer: Tokenizer for query processing
        use_reranker: Whether to apply LTR reranking
        candidate_topn: Number of candidates to retrieve for reranking
        final_topk: Final number of results to return
    
    Returns:
        Tuple of (documents, final_query, time_start)
    """
    if splade_encoder is not None:
        query = " ".join(splade_encoder.expand(query))

    time_start = time.perf_counter()
    
    if use_reranker and reranker is not None:
        # Use LTR reranker which handles retrieval + reranking
        documents = reranker.rerank(query, candidate_topn=candidate_topn, topk=final_topk)
    else:
        # Use traditional BM25 retrieval
        query_listed = tokenizer.tokenize(query)
        documents = retriever(query_listed)
    
    return documents, query, time_start


@hydra.main(config_name="search_ltr", version_base=None, config_path="../../../configs")
def main(cfg):
    print("Config:")
    print(cfg)
    history = InMemoryHistory()
    session = PromptSession(history=history)
    
    # Initialize base components
    bm25_ranking = BM25Ranking(cfg)
    ranker = BM25(bm25_ranking, cfg=cfg)
    tokenizer = get_tokenizer(cfg)
    chunker = Chunker(cfg=cfg)
    
    # Initialize SPLADE encoder if enabled
    splade_encoder = None
    if cfg.SEARCH.EXPAND_QUERIES:
        print("Loading SPLADE encoder for query expansion...")
        splade_encoder = SpladeEncoder(cfg=cfg)
    
    # Initialize LTR reranker if enabled
    reranker = None
    use_reranker = False
    candidate_topn = cfg.SEARCH.MAX_RESULTS
    
    if cfg.SEARCH.RERANK.ACTIVE:
        model_path = Path(cfg.SEARCH.RERANK.MODEL_PATH)
        if model_path.exists():
            print(f"Loading LTR reranker from {model_path}...")
            try:
                from sea.ltr.serve_tfr import TFRReranker
                reranker = TFRReranker.load(model_path=model_path, cfg=cfg)
                use_reranker = True
                candidate_topn = cfg.SEARCH.RERANK.CANDIDATE_TOPN
                print("LTR reranker loaded successfully!")
            except Exception as e:
                print(f"Warning: Could not load LTR reranker: {e}")
                print("Falling back to BM25 only.")
        else:
            print(f"Warning: Model path {model_path} does not exist.")
            print("Falling back to BM25 only.")
    
    # Display configuration
    print("\n" + "="*60)
    print("Search Configuration:")
    print(f"  SPLADE Query Expansion: {'Enabled' if splade_encoder else 'Disabled'}")
    print(f"  LTR Reranking: {'Enabled' if use_reranker else 'Disabled'}")
    if use_reranker:
        print(f"  Candidate Pool Size: {candidate_topn}")
    print(f"  Final Results: {cfg.SEARCH.MAX_RESULTS}")
    print("="*60 + "\n")

    while True:
        try:
            query = session.prompt("\nEnter your search query (or 'quit' to exit):\n> ")
        except EOFError:
            break  # Exit on Ctrl-D

        if query.lower() == 'quit':
            break
        if not query:
            continue

        documents, final_query, t0 = search_documents(
            retriever=ranker,
            reranker=reranker,
            query=query,
            splade_encoder=splade_encoder,
            tokenizer=tokenizer,
            use_reranker=use_reranker,
            candidate_topn=candidate_topn,
            final_topk=cfg.SEARCH.MAX_RESULTS
        )
        
        elapsed = (time.perf_counter() - t0) * cfg.SEARCH.TIME_MULTIPLIER
        history.append_string(query)
        chunker.set_query(final_query.split())

        if documents:
            for doc in documents:
                doc.pprint(verbose=cfg.SEARCH.VERBOSE_OUTPUT, loud=True, chunker=chunker)

            print(f"\nFound {len(documents)} matches in {elapsed:.2f} milliseconds.")
        else:
            print("\nNo matches found.")
        
        print(f"Query: {final_query}")
        if use_reranker:
            print(f"(Retrieved {candidate_topn} candidates, reranked to top {len(documents)})")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
        sys.exit(0)

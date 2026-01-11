import sys
import time

from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory

from sea.index.tokenization import get_tokenizer
from sea.ranking.io_wrapper import bm25
from sea.utils.chunker import Chunker
from sea.utils.config_wrapper import Config


def main():
    cfg = Config(load=True)
    history = InMemoryHistory()
    session = PromptSession(history=history)
    chunker = Chunker(cfg=cfg)
    tokenizer = get_tokenizer(cfg)

    retrieval_method = cfg.SEARCH.RETRIEVAL  # "bm25" or "semantic"
    reranker_enabled = getattr(cfg.SEARCH.RERANKER, "ENABLED", False)
    topk = cfg.SEARCH.MAX_RESULTS
    verbose = cfg.SEARCH.VERBOSE_OUTPUT

    # Build retriever based on config
    retriever = None
    semantic_searcher = None
    bm25_retriever = None
    reranker = None
    splade_encoder = None

    if retrieval_method == "semantic":
        from sea.ltr.bm25 import BM25Retriever
        from sea.semantic.search import SemanticSearcher

        semantic_searcher = SemanticSearcher(cfg=cfg, verbose=verbose)
        bm25_retriever = BM25Retriever.from_config(cfg)
        strategy = "Semantic"

    else:  # bm25
        retriever = bm25(cfg)
        strategy = "BM25"

        if cfg.SEARCH.EXPAND_QUERIES:
            from sea.query.splade import SpladeEncoder

            splade_encoder = SpladeEncoder(cfg=cfg)
            strategy += " + SPLADE"

    # Optional LTR reranker
    if reranker_enabled:
        model_path = cfg.SEARCH.RERANKER.MODEL_PATH
        candidate_topn = cfg.SEARCH.RERANKER.CANDIDATE_TOPN
        from sea.ltr.serve_tfr import TFRReranker

        reranker = TFRReranker.load(model_path=model_path, cfg=cfg)
        strategy += " + LTR Reranker"

    print(f"Search strategy: {strategy}")

    while True:
        try:
            query = session.prompt("\nEnter your search query (or 'quit' to exit):\n> ")
        except EOFError:
            break

        if query.lower() == "quit":
            break
        if not query:
            continue

        t0 = time.perf_counter()
        final_query = query

        # Apply SPLADE expansion if enabled (BM25 only)
        if splade_encoder is not None:
            final_query = " ".join(splade_encoder.expand(query))

        # Retrieval
        if reranker is not None:
            # Reranker handles retrieval internally
            documents = reranker.rerank(final_query, candidate_topn=candidate_topn, topk=topk)

        elif semantic_searcher is not None:
            results = semantic_searcher.search(query, topn=topk)
            documents = bm25_retriever.hydrate_docs(results)

        else:
            # BM25 retrieval
            tokens = tokenizer.tokenize(final_query)
            documents = retriever(tokens)

        elapsed = (time.perf_counter() - t0) * 1000
        history.append_string(query)
        chunker.set_query(final_query.split())

        if documents:
            for doc in documents:
                doc.pprint(verbose=verbose, loud=True, chunker=chunker)
            print(f"\nFound {len(documents)} matches in {elapsed:.2f} ms")
        else:
            print("\nNo matches found.")

        if final_query != query:
            print(f"Expanded query: {final_query}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting.")
        sys.exit(0)

import sys
import time

from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory

from sea.index.tokenization import get_tokenizer
from sea.ranking.io_wrapper import bm25
from sea.utils.chunker import Chunker
from sea.utils.config_wrapper import Config


def build_search_components(cfg):
    """Build search components based on configuration.

    Returns a dict with the initialized components and strategy string.
    """
    components = {
        "retriever": None,
        "semantic_searcher": None,
        "bm25_retriever": None,
        "reranker": None,
        "splade_encoder": None,
        "candidate_topn": None,
    }
    strategy_parts = []

    if cfg.SEARCH.RETRIEVAL == "semantic":
        from sea.ltr.bm25 import BM25Retriever
        from sea.semantic.search import SemanticSearcher

        components["semantic_searcher"] = SemanticSearcher(cfg=cfg, verbose=cfg.SEARCH.VERBOSE_OUTPUT)
        components["bm25_retriever"] = BM25Retriever.from_config(cfg)
        strategy_parts.append("Semantic")
    else:
        components["retriever"] = bm25(cfg)
        strategy_parts.append("BM25")

        if cfg.SEARCH.EXPAND_QUERIES:
            from sea.query.splade import SpladeEncoder

            components["splade_encoder"] = SpladeEncoder(cfg=cfg)
            strategy_parts.append("SPLADE")

    if getattr(cfg.SEARCH.RERANKER, "ENABLED", False):
        from sea.ltr.serve_tfr import TFRReranker

        components["reranker"] = TFRReranker.load(model_path=cfg.SEARCH.RERANKER.MODEL_PATH, cfg=cfg)
        components["candidate_topn"] = cfg.SEARCH.RERANKER.CANDIDATE_TOPN
        strategy_parts.append("LTR Reranker")

    components["strategy"] = " + ".join(strategy_parts)
    return components


def execute_search(query, components, tokenizer, cfg):
    """Execute search using the configured components.

    Returns list of Document results.
    """
    final_query = query

    if components["splade_encoder"] is not None:
        final_query = " ".join(components["splade_encoder"].expand(query))

    if components["reranker"] is not None:
        return components["reranker"].rerank(
            final_query,
            candidate_topn=components["candidate_topn"],
            topk=cfg.SEARCH.MAX_RESULTS
        ), final_query

    if components["semantic_searcher"] is not None:
        results = components["semantic_searcher"].search(query, topn=cfg.SEARCH.MAX_RESULTS)
        return components["bm25_retriever"].hydrate_docs(results), final_query

    tokens = tokenizer.tokenize(final_query)
    return components["retriever"](tokens), final_query


def main():
    cfg = Config(load=True)
    history = InMemoryHistory()
    session = PromptSession(history=history)
    chunker = Chunker(cfg=cfg)
    tokenizer = get_tokenizer(cfg)

    components = build_search_components(cfg)
    print(f"Search strategy: {components['strategy']}")

    verbose = cfg.SEARCH.VERBOSE_OUTPUT

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
        documents, final_query = execute_search(query, components, tokenizer, cfg)
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

import sys
import time

import numpy as np
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from rich.console import Console

from sea.index.tokenization import get_tokenizer
from sea.ranking.io_wrapper import bm25
from sea.utils.chunker import Chunker
from sea.utils.config_wrapper import Config

console = Console()

SEARCH_MODES = [
    ("BM25", "keyword matching", "bm25", False),
    ("BM25 + LTR", "keyword matching with reranker", "bm25", True),
    ("Semantic + LTR", "vector similarity with reranker", "semantic", True),
]


def print_header() -> None:
    console.print()
    console.print("╔══════════════════════════════════════╗", style="bold")
    console.print("║         SEA Search Engine            ║", style="bold")
    console.print("╚══════════════════════════════════════╝", style="bold")
    console.print()


def select_mode() -> tuple[str, bool]:
    console.print("Select search mode:\n", style="bold")

    for i, (name, desc, _, _) in enumerate(SEARCH_MODES, 1):
        console.print(f"  [{i}] {name}", style="bold white", end="")
        console.print(f"  ({desc})", style="dim")

    console.print()

    num_modes = len(SEARCH_MODES)
    while True:
        try:
            choice = input(f"Enter choice (1-{num_modes}): ").strip()
            if choice.isdigit() and 1 <= int(choice) <= num_modes:
                _, _, retrieval, use_reranker = SEARCH_MODES[int(choice) - 1]
                return retrieval, use_reranker
            console.print(f"Please enter a number between 1 and {num_modes}.", style="yellow")
        except (EOFError, KeyboardInterrupt):
            console.print("\nExiting.", style="dim")
            sys.exit(0)


def print_status(message: str, success: bool = True) -> None:
    indicator, color = ("✓", "green") if success else ("✗", "red")
    console.print(f"  [{indicator}]", style=f"bold {color}", end="")
    console.print(f" {message}")


def build_search_components(cfg, retrieval: str, use_reranker: bool) -> dict:
    components = {
        "retriever": None,
        "semantic_searcher": None,
        "bm25_retriever": None,
        "reranker": None,
        "splade_encoder": None,
        "candidate_topn": None,
        "doc_count": 0,
    }
    strategy_parts = []

    console.print("\nLoading components...", style="bold")

    if retrieval == "semantic":
        try:
            from sea.ltr.bm25 import BM25Retriever
            from sea.semantic.search import SemanticSearcher

            components["semantic_searcher"] = SemanticSearcher(cfg=cfg, verbose=False)
            components["bm25_retriever"] = BM25Retriever.from_config(cfg)
            components["doc_count"] = components["semantic_searcher"].corpus.shape[0]
            print_status(f"Semantic index loaded ({components['doc_count']:,} documents)")
            print_status("Embedding service connected")
            strategy_parts.append("Semantic")
        except Exception as e:
            print_status(f"Failed to load semantic search: {e}", success=False)
            sys.exit(1)
    else:
        try:
            components["retriever"] = bm25(cfg)
            components["doc_count"] = cfg.SEARCH.NUM_DOCS
            print_status(f"BM25 index loaded ({components['doc_count']:,} documents)")
            strategy_parts.append("BM25")
        except Exception as e:
            print_status(f"Failed to load BM25 index: {e}", success=False)
            sys.exit(1)

        if cfg.SEARCH.EXPAND_QUERIES:
            try:
                from sea.query.splade import SpladeEncoder
                components["splade_encoder"] = SpladeEncoder(cfg=cfg)
                print_status("SPLADE query expansion ready")
                strategy_parts.append("SPLADE")
            except Exception as e:
                print_status(f"SPLADE expansion unavailable: {e}", success=False)

    if use_reranker:
        try:
            from sea.ltr.serve_tfr import TFRReranker
            components["reranker"] = TFRReranker.load(
                model_path=cfg.SEARCH.RERANKER.MODEL_PATH, cfg=cfg
            )
            components["candidate_topn"] = cfg.SEARCH.RERANKER.CANDIDATE_TOPN
            print_status("LTR reranker ready")
            strategy_parts.append("LTR")
        except Exception as e:
            print_status(f"Failed to load reranker: {e}", success=False)
            sys.exit(1)

    components["strategy"] = " + ".join(strategy_parts)

    console.print()
    console.print("Ready to search!", style="bold green")

    return components


def apply_ltr_reranking(query: str, docs: list, reranker, max_results: int) -> list:
    if not docs:
        return []

    features = reranker.fe.extract_many(query, docs)
    list_size = reranker.model.input_shape[1]
    num_features = reranker.model.input_shape[2]
    num_docs = features.shape[0]

    padded = np.zeros((1, list_size, num_features), dtype=np.float32)
    use_count = min(num_docs, list_size)
    padded[0, :use_count, :] = features[:use_count, :]

    scores = reranker.model.predict(padded, verbose=0)[0][:num_docs]
    ranked_indices = np.argsort(-scores)

    reranked = []
    for idx in ranked_indices[:max_results]:
        if idx < len(docs):
            docs[idx].score = float(scores[idx])
            reranked.append(docs[idx])
    return reranked


def execute_search(query: str, components: dict, tokenizer, cfg) -> tuple[list, str]:
    final_query = query

    if components["splade_encoder"] is not None:
        final_query = " ".join(components["splade_encoder"].expand(query))

    if components["semantic_searcher"] is not None and components["reranker"] is not None:
        results = components["semantic_searcher"].search(query, topn=components["candidate_topn"])
        docs = components["bm25_retriever"].hydrate_docs(results)
        reranked = apply_ltr_reranking(query, docs, components["reranker"], cfg.SEARCH.MAX_RESULTS)
        return reranked, final_query

    if components["reranker"] is not None:
        results = components["reranker"].rerank(
            final_query,
            candidate_topn=components["candidate_topn"],
            topk=cfg.SEARCH.MAX_RESULTS
        )
        return results, final_query

    tokens = tokenizer.tokenize(final_query)
    return components["retriever"](tokens), final_query


def print_search_header(mode: str) -> None:
    console.print()
    console.print("━" * 40, style="dim")
    console.print(f"Mode: {mode}", style="bold")
    console.print("━" * 40, style="dim")


def print_results(documents: list, elapsed_ms: float, chunker, verbose: bool) -> None:
    if not documents:
        console.print("\nNo matches found.", style="yellow")
        return

    for rank, doc in enumerate(documents, 1):
        console.print()
        console.print("━" * 40, style="dim")
        doc.pprint(verbose=verbose, loud=True, chunker=chunker, rank=rank)

    console.print()
    console.print("━" * 40, style="dim")
    console.print(f"\nFound {len(documents)} results in {elapsed_ms:.1f}ms", style="bold")


def main() -> None:
    cfg = Config(load=True)

    # Show header and mode selection
    print_header()
    retrieval, use_reranker = select_mode()

    # Build components with status display
    components = build_search_components(cfg, retrieval, use_reranker)

    # Initialize session
    history = InMemoryHistory()
    session = PromptSession(history=history)
    chunker = Chunker(cfg=cfg)
    tokenizer = get_tokenizer(cfg)
    verbose = cfg.SEARCH.VERBOSE_OUTPUT

    # Search loop
    while True:
        print_search_header(components["strategy"])

        try:
            query = session.prompt("> ")
        except EOFError:
            break

        if query.lower() == "quit":
            break
        if not query.strip():
            continue

        t0 = time.perf_counter()
        documents, final_query = execute_search(query, components, tokenizer, cfg)
        elapsed = (time.perf_counter() - t0) * 1000

        history.append_string(query)
        chunker.set_query(final_query.split())

        print_results(documents, elapsed, chunker, verbose)

        if final_query != query:
            console.print(f"Expanded query: {final_query}", style="dim italic")

    console.print("\nGoodbye!", style="dim")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\nExiting.", style="dim")
        sys.exit(0)

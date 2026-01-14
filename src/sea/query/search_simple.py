import sys
import time

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


def print_header():
    """Display the application header."""
    console.print()
    console.print("╔══════════════════════════════════════╗", style="bold")
    console.print("║         SEA Search Engine            ║", style="bold")
    console.print("╚══════════════════════════════════════╝", style="bold")
    console.print()


def select_mode() -> tuple[str, bool]:
    """Display mode selection menu and return (retrieval_type, use_reranker)."""
    console.print("Select search mode:\n", style="bold")

    num_modes = len(SEARCH_MODES)
    for i, (name, desc, _, _) in enumerate(SEARCH_MODES, 1):
        console.print(f"  [{i}] {name}", style="bold white", end="")
        console.print(f"  ({desc})", style="dim")

    console.print()

    valid_choices = [str(i) for i in range(1, num_modes + 1)]
    while True:
        try:
            choice = input(f"Enter choice (1-{num_modes}): ").strip()
            if choice in valid_choices:
                idx = int(choice) - 1
                _, _, retrieval, use_reranker = SEARCH_MODES[idx]
                return retrieval, use_reranker
            console.print(f"Please enter a number between 1 and {num_modes}.", style="yellow")
        except (EOFError, KeyboardInterrupt):
            console.print("\nExiting.", style="dim")
            sys.exit(0)


def print_status(message: str, success: bool = True):
    """Print a status line with colored indicator."""
    indicator = "✓" if success else "✗"
    color = "green" if success else "red"
    console.print(f"  [{indicator}]", style=f"bold {color}", end="")
    console.print(f" {message}")


def build_search_components(cfg, retrieval: str, use_reranker: bool):
    """Build search components based on selected mode.

    Returns a dict with the initialized components and strategy string.
    """
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


def execute_search(query, components, tokenizer, cfg):
    """Execute search using the configured components.

    Returns list of Document results.
    """
    import numpy as np

    final_query = query

    if components["splade_encoder"] is not None:
        final_query = " ".join(components["splade_encoder"].expand(query))

    # Semantic + LTR: use semantic search for candidates, then rerank with LTR
    if components["semantic_searcher"] is not None and components["reranker"] is not None:
        results = components["semantic_searcher"].search(query, topn=components["candidate_topn"])
        docs = components["bm25_retriever"].hydrate_docs(results)
        if not docs:
            return [], final_query

        # Extract features and apply LTR model
        reranker = components["reranker"]
        X = reranker.fe.extract_many(query, docs)
        expected_list_size = reranker.model.input_shape[1]
        num_features = reranker.model.input_shape[2]
        num_docs = X.shape[0]

        X_padded = np.zeros((1, expected_list_size, num_features), dtype=np.float32)
        use_count = min(num_docs, expected_list_size)
        X_padded[0, :use_count, :] = X[:use_count, :]

        scores = reranker.model.predict(X_padded, verbose=0)[0]
        actual_scores = scores[:num_docs]
        order = np.argsort(-actual_scores)

        reranked = []
        for i in order[:cfg.SEARCH.MAX_RESULTS]:
            if i >= len(docs):
                continue
            d = docs[int(i)]
            d.score = float(actual_scores[int(i)])
            reranked.append(d)
        return reranked, final_query

    # BM25 + LTR: use BM25 for candidates, then rerank with LTR
    if components["reranker"] is not None:
        return components["reranker"].rerank(
            final_query,
            candidate_topn=components["candidate_topn"],
            topk=cfg.SEARCH.MAX_RESULTS
        ), final_query

    # BM25 only
    tokens = tokenizer.tokenize(final_query)
    return components["retriever"](tokens), final_query


def print_search_header(mode: str):
    """Print the search prompt header."""
    console.print()
    console.print("━" * 40, style="dim")
    console.print(f"Mode: {mode}", style="bold")
    console.print("━" * 40, style="dim")


def print_results(documents, elapsed_ms: float, chunker, verbose: bool):
    """Print search results with rank numbers and separators."""
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


def main():
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

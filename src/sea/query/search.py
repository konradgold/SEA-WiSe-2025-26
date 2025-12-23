import sys
import time

from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory

from sea.index.tokenization import get_tokenizer
from sea.query.parser import QueryParser
from sea.ranking import RankersRegistry
from sea.utils.config import Config


def search_documents(
    *,
    cfg: Config,
    ranker,
    query: str,
    splade_encoder,
    max_output_result: int,
    tokenizer,
):
    # This assumes documents are stored with keys like 'D*' (e.g., 'D1972382') and contain JSON content
    query_parser = QueryParser(cfg=cfg)
    root_operator = query_parser.process_phrase2query(query, splade_encoder=splade_encoder)
    matches, final_query = root_operator.execute(ranker, tokenizer)
    return (sorted(matches)[:max_output_result], len(matches), final_query) if matches else (None, 0, final_query)


def main():
    cfg = Config(load=True)
    history = InMemoryHistory()
    session = PromptSession(history=history)
    max_output_result = cfg.SEARCH.MAX_RESULTS if cfg.SEARCH.MAX_RESULTS is not None else 10
    splade_encoder = None
    if cfg.SEARCH.EXPAND_QUERIES:
        from sea.query.splade import SpladeEncoder
        splade_encoder = SpladeEncoder(cfg=cfg)

    tokenizer = get_tokenizer(cfg)
    ranker_builder = RankersRegistry.get_ranker(cfg.SEARCH.RANKING)
    ranker = ranker_builder()
    while True:
        try:
            query = session.prompt("\nEnter your search query (or 'quit' to exit):\n> ")
        except EOFError:
            break  # Exit on Ctrl-D

        if query.lower() == 'quit':
            break
        if not query:
            continue   

        t0 = time.time()    
        documents, num_matches, final_query = search_documents(
            cfg=cfg,
            ranker=ranker,
            query=query,
            splade_encoder=splade_encoder,
            max_output_result=max_output_result,
            tokenizer=tokenizer,
        )
        elapsed = (time.time() - t0)*1000
        history.append_string(final_query)

        if documents:
            for doc in documents:
                doc.pprint(verbose=cfg.SEARCH.VERBOSE_OUTPUT, loud=True)

            print(f"\nFound {num_matches} matches in {elapsed:.2f} milliseconds.")
        else:
            print("\nNo matches found.")
        print(f"Query: {final_query}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
        sys.exit(0)

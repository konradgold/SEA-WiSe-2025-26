import json
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
import sys
from sea.index.tokenization import get_tokenizer
from sea.perf.simple_perf import perf_indicator
from sea.query import splade
from sea.query.parser import QueryParser
from sea.storage.interface import get_storage
from sea.utils.config import Config
import time
from sea.ranking.utils import Document


@perf_indicator("search", "queries")
def search_documents(redis_client, query, max_output_result=10, tokenizer=None):
    # This assumes documents are stored with keys like 'D*' (e.g., 'D1972382') and contain JSON content
    cfg = Config(load=True)
    query_parser = QueryParser(cfg=cfg)
    root_operator = query_parser.process_phrase2query(query)
    matches = root_operator.execute(redis_client, tokenizer)
    if not matches:
        return []
    num_matches = len(matches)
    out_matches = []
    for i, match in enumerate(matches):
        if i >= max_output_result:
            break
        doc_content = redis_client.get(match)
        if doc_content:
            doc_json = json.loads(doc_content)
            out_matches.append(Document(
                doc_id=match,
                title=doc_json.get("title", ""),
                link=doc_json.get("link", ""),
                content=doc_json.get("content", None)
            ))
    return out_matches, num_matches

def main():
    cfg = Config(load=True)
    cfg.STORAGE.LOAD_DOCUMENTS = True
    client = get_storage(cfg)
    history = InMemoryHistory()
    session = PromptSession(history=history)
    max_output_result = cfg.SEARCH.MAX_RESULTS if cfg.SEARCH.MAX_RESULTS is not None else 10
    splade_encoder = None
    if cfg.SEARCH.EXPAND_QUERIES:
        from sea.query.splade import SpladeEncoder
        splade_encoder = SpladeEncoder(cfg=cfg)
    if cfg.SEARCH.RANKING is not None:
        from sea.ranking import RankersRegistry
        ranker_builder = RankersRegistry.get_ranker(cfg.SEARCH.RANKING)
        ranker = ranker_builder()
    tokenizer = get_tokenizer()

    while True:
        try:
            query = session.prompt("\nEnter your search query (or 'quit' to exit):\n> ")
        except EOFError:
            break  # Exit on Ctrl-D

        if query.lower() == 'quit':
            break
        if not query:
            continue

        if splade_encoder:
            t_exp = time.time()
            expansion_tokens = splade_encoder.expand(query)
            elapsed_exp = (time.time() - t_exp)*1000
            print(f"Expansion took {elapsed_exp:.2f} milliseconds.")
            if expansion_tokens:
                print(f"Expanded query tokens: {expansion_tokens}")
                query = " ".join(expansion_tokens)
                print(query)

        history.append_string(query)
        t0 = time.time()
        if cfg.SEARCH.RANKING is not None:
            t_tok = time.time()
            tokens = tokenizer.tokenize(query)
            elapsed_tok = (time.time() - t_tok)*1000
            print(f"Tokenization took {elapsed_tok:.2f} milliseconds.")
            t0 = time.time()
            documents = ranker(tokens)  # type: ignore
            num_matches = len(documents)
        else:
            documents, num_matches = search_documents(client, query, max_output_result, tokenizer)
        elapsed = (time.time() - t0)*1000

        if documents:
            for doc in documents:
                doc.pprint(verbose=cfg.SEARCH.VERBOSE_OUTPUT, loud=True)

            print(f"\nFound {num_matches} matches in {elapsed:.2f} milliseconds.")
        else:
            print("\nNo matches found.")
        print(f"Query: {query}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
        sys.exit(0)

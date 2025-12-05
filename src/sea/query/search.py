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



@perf_indicator("search", "queries")
def search_documents(redis_client, query, max_output_result=10):
    # This assumes documents are stored with keys like 'D*' (e.g., 'D1972382') and contain JSON content
    query_parser = QueryParser(cfg=Config())
    root_operator = query_parser.process_phrase2query(query)
    matches = root_operator.execute(redis_client, get_tokenizer())
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
            out_matches.append((match, doc_json.get("title", ""), doc_json.get("link", "")))
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
            expansion_tokens = splade_encoder.expand(query)
            if expansion_tokens:
                print(f"Expanded query tokens: {expansion_tokens}")
                query = " or ".join(expansion_tokens)
                print(query)

        history.append_string(query)
        t0 = time.time()
        results, num_matches = search_documents(client, query, max_output_result)
        elapsed = (time.time() - t0)*1000

        if results:
            for key, title, link in results:
                print(f"\n{key}:")
                print(title)
                print(link)

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

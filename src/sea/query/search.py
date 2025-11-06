import json
import redis
import sys
from sea.index.tokenization import get_tokenizer
from sea.perf.simple_perf import perf_indicator
from sea.query.parser import QueryParser
from sea.utils.config import Config
from sea.utils.manage_redis import connect_to_db


@perf_indicator("search", "queries")
def search_documents(redis_client, query):
    # This assumes documents are stored with keys like 'D*' (e.g., 'D1972382') and contain JSON content
    query_parser = QueryParser(cfg=Config())
    root_operator = query_parser.process_phrase2query(query)
    matches = root_operator.execute(redis_client, get_tokenizer())
    if not matches:
        return []
    out_matches = []
    for match in matches:
        doc_content = redis_client.get(match)
        if doc_content:
            doc_json = json.loads(doc_content)
            out_matches.append((match, doc_json.get("title", ""), doc_json.get("link", "")))
    return out_matches

def main():
    cfg = Config(load=True)
    redis_client = connect_to_db(cfg)
    while True:
        print("\nEnter your search query (or 'quit' to exit):")
        query = input("> ")
        if query.lower() == 'quit':
            break
        if not query:
            continue

        results = search_documents(redis_client, query)

        if results:
            for key, title, link in results:
                print(f"\n{key}:")
                print(title)
                print(link)
            print(f"\nFound {len(results)} matches")
        else:
            print("\nNo matches found.")
        print(f"Query: {query}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
        sys.exit(0)

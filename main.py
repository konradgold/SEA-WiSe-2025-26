import json
import redis
import sys
import os
from transformers import AutoTokenizer
from redis.commands.json.path import Path
from perf.simple_perf import perf_indicator
from utils.config import Config


def connect_to_redis(host='localhost', port=6379):
    try:
        # Connect to Redis (adjust host/port if needed)
        r = redis.Redis(host=host, port=port, decode_responses=True)
        r.ping()  # Test connection
        return r
    except redis.ConnectionError:
        print("Error: Could not connect to Redis. Make sure Redis server is running.")
        sys.exit(1)


@perf_indicator("search", "queries")
def search_documents(redis_client, query):
    # This assumes documents are stored with keys like 'doc:1', 'doc:2', etc.
    # and contain text content
    matches = None  
    keys = ["token:" + str(token) for token in query]
    for key in keys:
        content = redis_client.json().get(key, Path.root_path())
        doc_keys = set(content.get("documents", {}).keys()) if content else set()
        if matches is None:
            matches = set(doc_keys)
        else:
            matches.intersection_update(doc_keys)
        if matches is not None and len(matches) == 0:
            # AND means no result if any token has no matches
            break
    if not matches:
        return []
    out_matches = []
    for match in matches:
        doc_content = redis_client.get(match)
        if doc_content:
            doc_json = json.loads(doc_content)
            out_matches.append(
                (match, doc_json.get("title", ""), doc_json.get("link", ""))
            )
    return out_matches

def main():
    cfg = Config(load=True)
    redis_client = connect_to_redis(cfg.REDIS_HOST, cfg.REDIS_PORT)
    tokenizer = AutoTokenizer.from_pretrained(cfg.TOKENIZER.BACKEND)
    
    while True:
        print("\nEnter your search query (or 'quit' to exit):")
        query = input("> ")
        if query.lower() == 'quit':
            break
        if not query:
            continue

        query = tokenizer.encode(query)
        
        
        results = search_documents(redis_client, query)
        
        if results:
            for key, title, link in results:
                print(f"\n{key}:")
                print(title)
                print(link)
            print(f"\nFound {len(results)} matches")
        else:
            print("\nNo matches found.")
        print(f"Tokenized query: {query}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
        sys.exit(0)

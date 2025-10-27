import json
import redis
import sys
import os
from tokenization import get_tokenizer
from perf.simple_perf import perf_indicator


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
    # This assumes documents are stored with keys like 'D*' (e.g., 'D1972382') and contain JSON content
    matches = None  
    keys = ["token:" + str(token) for token in query]
    for key in keys:
        # Hash postings: field = doc_id, value = tf
        postings = redis_client.hgetall(key)
        if not postings:
            # AND semantics: if any token has no postings, result is empty
            matches = set()
            break
        doc_ids = set(postings.keys())
        if matches is None:
            matches = doc_ids
        else:
            matches.intersection_update(doc_ids)
        if not matches:
            # Early exit if intersection is empty
            break
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
    redis_client = connect_to_redis(os.getenv("REDIS_HOST", "localhost"), int(os.getenv("REDIS_PORT", 6379)))
    tokenizer = get_tokenizer()

    while True:
        print("\nEnter your search query (or 'quit' to exit):")
        query = input("> ")
        if query.lower() == 'quit':
            break
        if not query:
            continue

        query = tokenizer.tokenize(query)

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

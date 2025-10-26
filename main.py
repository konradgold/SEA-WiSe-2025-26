import json
import redis
import sys
import os
from tokenization import get_tokenizer


def connect_to_redis(host='localhost', port=6379):
    try:
        # Connect to Redis (adjust host/port if needed)
        r = redis.Redis(host=host, port=port, decode_responses=True)
        r.ping()  # Test connection
        return r
    except redis.ConnectionError:
        print("Error: Could not connect to Redis. Make sure Redis server is running.")
        sys.exit(1)

def search_documents(redis_client, query):
    # This assumes documents are stored with keys like 'doc:1', 'doc:2', etc.
    # and contain text content
    matches = set()
    keys = ["token:" + str(token) for token in query]
    for key in  keys:
        print(key)
        # Hash postings: field = doc_id, value = tf
        postings = redis_client.hgetall(key)
        if not postings:
            # No postings list for this token; intersect to empty set
            matches = set() if matches else matches
            continue
        matches.intersection_update(postings.keys()) if matches else matches.update(postings.keys())
    if matches:
        out_matches = []
        for match in matches:
            doc_content = redis_client.get(match)
            if doc_content:
                doc_json = json.loads(doc_content)
                out_matches.append((match, doc_json.get("title", ""), doc_json.get("link", "")))
        return out_matches

 


def main():
    redis_client = connect_to_redis(os.getenv("REDIS_HOST", "localhost"), int(os.getenv("REDIS_PORT", 6379)))
    tokenizer = get_tokenizer(os.getenv("TOKENIZER_BACKEND"))
    
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
import json
import redis
import sys
from dotenv import load_dotenv
import os


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
    matches = []
    for key in redis_client.keys():
        content = redis_client.get(key)
        content_json = json.loads(content)
        if query.lower() in content_json.get("body", "").lower():
            matches.append((key, content_json["title"], content_json["link"]))
    return matches

def main():
    redis_client = connect_to_redis(os.getenv("REDIS_HOST", "localhost"), int(os.getenv("REDIS_PORT", 6379)))
    
    while True:
        print("\nEnter your search query (or 'quit' to exit):")
        query = input("> ").strip()
        
        if query.lower() == 'quit':
            break
        
        if not query:
            continue
            
        results = search_documents(redis_client, query)
        
        if results:
            print(f"\nFound {len(results)} matches:")
            for key, title, link in results:
                print(f"\n{key}:")
                print(title)
                print(link)
        else:
            print("\nNo matches found.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
        sys.exit(0)
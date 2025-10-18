import argparse
from dotenv import load_dotenv
import os
from transformers import GPT2Tokenizer
from redis.commands.json.path import Path
import redis
import json
from transformers import AutoTokenizer

def connect_to_db(host: str, port: int):
    # Placeholder for database connection logic
    return redis.Redis(host=host, port=port, decode_responses=True)

def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description='Tokenize documents in Redis')
    parser.add_argument('--redis-port', type=int, default=int(os.getenv("REDIS_PORT", 6379)),
                      help='Redis server port')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


    redis_port = args.redis_port

    db = connect_to_db("localhost", redis_port)
    pipe = db.pipeline()
    keys = [k for k in db.keys() if not k.startswith('token:')]
    for key in keys:
        try:
            content = db.get(key)
            content_json = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            print(f"Skipping key {key}: unable to decode JSON.")
            continue
        body = content_json.get("body", "")
        tokens = tokenizer.encode(body)
        content_json["tokens"] = tokens


        unique_tokens = set(tokens)
        for token in unique_tokens:
            token_key = f"token:{token}"
            ok = db.json().set(token_key, Path.root_path(), {"documents": {}}, nx=True)
        token_counts = {token: tokens.count(token) for token in unique_tokens}
        print(f"Initialized token counts for {len(unique_tokens)} unique tokens in document {key}.")
        pipe.set(key, json.dumps(content_json))
        for token in unique_tokens:
            token_key = f"token:{token}"

            doc_field = {key: token_counts[token]}  # ensure `key` is str
            pipe.json().merge(token_key, Path("$.documents"), doc_field)
    result = pipe.execute()
    print(f"Number of correct tokenizations: {sum(result)}")

    db.close()

main()
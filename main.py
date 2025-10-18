import argparse
from injestion import Injestion, MinimalProcessor
import redis
from dotenv import load_dotenv
import os



def connect_to_db(host: str, port: int):
    # Placeholder for database connection logic
    return redis.Redis(host=host, port=port)

def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description='Ingest documents into Redis')
    parser.add_argument('--batch-size', type=int, default=400,
                      help='number of documents to ingest (default: 400)')
    parser.add_argument('--documents-path', type=str, default=os.getenv("DOCUMENTS", "msmarco-docs.tsv"),
                      help='path to the documents file')
    parser.add_argument('--redis-port', type=int, default=int(os.getenv("REDIS_PORT", 6379)),
                      help='Redis server port')
    args = parser.parse_args()

    documents_path = args.documents_path
    redis_port = args.redis_port

    db = connect_to_db("localhost", redis_port)
    injestion = Injestion(db, [MinimalProcessor()], documents_path)
    injestion.injest(args.batch_size)

main()

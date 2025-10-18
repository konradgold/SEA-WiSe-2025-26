import enum
import json
from typing import Any
import argparse
import redis
from dotenv import load_dotenv
import os

class Columns(enum.Enum):
    doc_id = "doc_id"
    link = "link"
    title = "title"
    body = "body"

class Processor:
    def process(self, id, document) -> tuple[str, Any]:
        # Placeholder for processing logic
        return id, document
    
class MinimalProcessor(Processor):
    def process(self, id, document) -> tuple[str, str]:
        return document[0], json.dumps({
            Columns.doc_id.value: document[0],
            Columns.link.value: document[1],
            Columns.title.value: document[2],
            Columns.body.value: document[3],
        })

class Injestion:
    """
    Injest documents into the database
    Idea: Chain processors that are parsed via list
    """
    def __init__(self, db, processors: list[Processor], document_path:str):

        self.db = db
        self.processors = processors
        self.document_path = document_path

    def injest(self, batch_size: int = 1000):
        pipeline = self.db.pipeline()
        with open(self.document_path, "r") as f:
            batch_count = 0
            for line in f:
                doc = line.strip().split("\t")
                doc_id = doc[0]
                if len(doc) != 4:
                    continue
                for processor in self.processors:
                    doc_id, doc = processor.process(id, doc)
                pipeline.setnx(doc_id, doc)
                batch_count += 1

                if batch_count >= batch_size:
                    results = pipeline.execute()
                    batch_count = 0
                    if sum(results) == batch_size:
                        print(f"Injested {sum(results)} documents")
                        break
                    else:
                        batch_size -= sum(results)
                        print(f"Partial ingestion: {sum(results)} documents")

        if batch_count > 0:
            pipeline.execute()
        print(f"There are now {self.db.dbsize()} documents in the database.")

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
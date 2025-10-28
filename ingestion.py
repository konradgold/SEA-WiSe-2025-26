import enum
import json
from typing import Any
import argparse
import redis
from dotenv import load_dotenv
import os
from perf.simple_perf import perf_indicator

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

class Ingestion:
    """
    Ingest documents into the database
    Idea: Chain processors that are parsed via list
    """
    def __init__(self, db, processors: list[Processor], document_path:str):

        self.db = db
        self.processors = processors
        self.document_path = document_path

    @perf_indicator("ingest", "docs")
    def ingest(self, batch_size: int = 1000):
        pipeline = self.db.pipeline()
        inserted_keys: list[str] = []
        keys_in_batch: list[str] = []
        remaining = batch_size
        total_inserted = 0

        with open(self.document_path, "r") as f:
            batch_count = 0
            for line in f:
                if remaining <= 0:
                    break
                doc = line.strip().split("\t")
                if len(doc) != 4:
                    continue
                doc_id = doc[0]
                for processor in self.processors:
                    doc_id, doc = processor.process(doc_id, doc)
                pipeline.setnx(doc_id, doc)
                keys_in_batch.append(doc_id)
                batch_count += 1

                if batch_count >= remaining:
                    results = pipeline.execute()
                    # Track which keys were newly inserted
                    for k, r in zip(keys_in_batch, results):
                        if bool(r):
                            inserted_keys.append(k)
                    ingested_now = sum(int(bool(r)) for r in results)
                    total_inserted += ingested_now
                    remaining -= ingested_now
                    batch_count = 0
                    keys_in_batch = []
                    if ingested_now == batch_size:
                        print(f"Injested {ingested_now} documents")
                    else:
                        print(f"Partial ingestion: {ingested_now} documents")

        if remaining > 0 and batch_count > 0:
            results = pipeline.execute()
            for k, r in zip(keys_in_batch, results):
                if bool(r):
                    inserted_keys.append(k)
            ingested_now = sum(int(bool(r)) for r in results)
            total_inserted += ingested_now

        print(f"There are now {self.db.dbsize()} documents in the database.")
        # Return (payload, count) for perf_indicator
        return inserted_keys, total_inserted

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
    ingestion = Ingestion(db, [MinimalProcessor()], documents_path)
    ingestion.ingest(args.batch_size)
    db.close()

if __name__ == "__main__":
    main()

import enum
import json
from typing import Any
import argparse
import redis
from dotenv import load_dotenv
import os
from perf.simple_perf import perf_indicator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    def ingest(self, num_documents: int = 1000, batch_size: int = 500):
        pipeline = self.db.pipeline()
        inserted_keys: list[str] = []
        keys_in_batch: list[str] = []
        remaining = num_documents
        if batch_size > num_documents or batch_size <= 0:
            batch_size = num_documents
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

                if batch_count >= batch_size:
                    total_inserted = self._execute_pipe(pipeline, keys_in_batch, inserted_keys, total_inserted)
                    remaining = num_documents - total_inserted
                    batch_count = 0
                    keys_in_batch = []
                    logger.info(f"{total_inserted} of {num_documents} documents ingested so far...")
                if remaining <= batch_count:
                    break

        if remaining > 0 and batch_count > 0:
            total_inserted = self._execute_pipe(pipeline, keys_in_batch, inserted_keys, total_inserted)

        logger.info(f"There are now {self.db.dbsize()} documents in the database.")
        # Return (payload, count) for perf_indicator
        return inserted_keys, total_inserted
    
    def _execute_pipe(self, pipeline, keys_in_batch, inserted_keys, total_inserted):
        results = pipeline.execute()
        # Track which keys were newly inserted
        for k, r in zip(keys_in_batch, results):
            if bool(r):
                inserted_keys.append(k)
        ingested_now = sum(results)
        total_inserted += ingested_now
        return total_inserted

def connect_to_db(host: str, port: int):
    # Placeholder for database connection logic
    return redis.Redis(host=host, port=port)

def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description='Ingest documents into Redis')
    parser.add_argument('--num-documents', type=int, default=400,
                      help='number of documents to ingest (default: 400)')
    parser.add_argument('--batch-size', type=int, default=100,
                      help='number of documents per batch (default: 100)')
    parser.add_argument('--documents-path', type=str, default=os.getenv("DOCUMENTS", "msmarco-docs.tsv"),
                      help='path to the documents file')
    parser.add_argument('--redis-port', type=int, default=int(os.getenv("REDIS_PORT", 6379)),
                      help='Redis server port')
    args = parser.parse_args()

    documents_path = args.documents_path
    redis_port = args.redis_port

    db = connect_to_db("localhost", redis_port)
    ingestion = Ingestion(db, [MinimalProcessor()], documents_path)
    ingestion.ingest(args.num_documents, args.batch_size)
    db.close()

if __name__ == "__main__":
    main()

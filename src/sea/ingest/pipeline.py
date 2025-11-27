import enum
import json
from typing import Any
import redis
from sea.index.tokenizer_job import process_batch_in_memory
from sea.perf.simple_perf import perf_indicator
import logging
from sea.storage.interface import get_storage
from sea.utils.config import Config

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
    # def __init__(self, db, processors: list[Processor], document_path:str):
    def __init__(self, processors: list[Processor], document_path:str):

        # self.db = db
        self.processors = processors
        self.document_path = document_path

    @perf_indicator("ingest", "docs")
    def ingest(self, num_documents: int = 1000, batch_size: int = 500):
        # pipeline = self.db.pipeline()
        inserted_keys: list[str] = []
        keys_in_batch: list[str] = []
        remaining = num_documents
        if batch_size > num_documents or batch_size <= 0:
            batch_size = num_documents
        total_inserted = 0
        total_visited = 0

        docs_metadata = dict()
        current_doc_no = 0

        batch = []

        print(f"Starting ingestion of {num_documents} documents from {self.document_path}...")
        with open(self.document_path, "r") as f:
            print("Opened document file.")
            batch_count = 0
            for index, line in enumerate(f):
                
                if remaining <= 0:
                    break
                doc = line.strip().split("\t")
                if len(doc) != 4:
                    continue

                docs_metadata[current_doc_no] = [doc[0], index] 
                doc[0] = current_doc_no
                doc_id = doc[0]


                doc_id = doc[0]
                for processor in self.processors:
                    doc_id, doc = processor.process(doc_id, doc)
                # pipeline.setnx(doc_id, doc)
                doc = json.loads(doc)
                # print(doc["body"])

                batch.append(doc)

                current_doc_no += 1
                remaining -= 1
                batch_count += 1

                if batch_count >= batch_size or remaining < 0:
                    # for multiprocessing in windows, we need to protect the entry point
                    if __name__ == "__main__":
                        index  = process_batch_in_memory(docs_metadata, batch)

                    print(docs_metadata)
                    batch = []
                    batch_count = 0

                    if remaining < 0:
                        break


  
  



        #         keys_in_batch.append(doc_id)
        #         batch_count += 1
        #         total_visited += 1

        #         if batch_count >= batch_size:
        #             total_inserted = self._execute_pipe(pipeline, keys_in_batch, inserted_keys, total_inserted)
        #             remaining = num_documents - total_inserted
        #             batch_count = 0
        #             keys_in_batch = []
        #             percent = (total_inserted / num_documents * 100) if num_documents else 0.0
        #             logger.info(
        #                 f"Ingested {total_inserted:,}/{num_documents:,} ({percent:.1f}%) — "
        #                 f"visited {total_visited:,}; remaining {remaining:,}"
        #             )
        #             if total_inserted == 0 and total_visited % 1000 == 0:
        #                 logger.info("This is intended behavior if documents already exist in the database.")
        #         if remaining <= batch_count:
        #             break

        # if remaining > 0 and batch_count > 0:
        #     total_inserted = self._execute_pipe(pipeline, keys_in_batch, inserted_keys, total_inserted)

        # logger.info(f"There are now {self.db.dbsize()} documents in the database.")
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


def main():
    cfg = Config(load=True)

    # cfg.STORAGE.KEEP_DOCUMENTS = True # otherwise this makes no sense
    # db = get_storage(cfg)
    ingestion = Ingestion([MinimalProcessor()], cfg.DOCUMENTS)
    # ingestion = Ingestion(db, [MinimalProcessor()], cfg.DOCUMENTS)
    ingestion.ingest(cfg.INGESTION.NUM_DOCUMENTS, cfg.INGESTION.BATCH_SIZE)
    # db.close()

if __name__ == "__main__":
    main()

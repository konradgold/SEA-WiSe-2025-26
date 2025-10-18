import enum
import json
from typing import Any

class Columns(enum.Enum):
    doc_id = "doc_id"
    link = "link"
    title = "title"
    body = "body"

class Processor:
    def process(self, document) -> tuple[str, Any]:
        # Placeholder for processing logic
        return document
    
class MinimalProcessor(Processor):
    def process(self, document) -> tuple[str, str]:
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
                if len(doc) != 4:
                    continue
                for processor in self.processors:
                    doc_id, doc = processor.process(doc)
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
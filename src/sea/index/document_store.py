"""
Document Store: Stores documents on disk with fast random access.

File format:
- documents.bin: Sequential document records
  [record_len(4B)][doc_id_int(4B)][url_len(2B)][url][title_len(2B)][title][body]
  
- doc_offsets.bin: Fixed-size offset table for O(1) lookup
  [num_docs(4B)][max_doc_id(4B)][(doc_id, offset) pairs sorted by doc_id]
  
This allows us to:
1. Store documents once during ingestion
2. Look up any document by ID in O(1) time during search
"""

import struct
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class DocumentStore:
    """
    Writes documents to disk during indexing, reads them during search.
    
    Documents are stored sequentially in documents.bin.
    A separate offsets file maps doc_id -> file offset for fast lookup.
    """
    
    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        self.docs_path = os.path.join(index_dir, "documents.bin")
        self.offsets_path = os.path.join(index_dir, "doc_offsets.bin")
        
        # In-memory offset table
        self.offsets: dict[int, int] = {}  # doc_id_int -> file_offset
        
        # File handles
        self._docs_file: Optional[object] = None
        self._mode: Optional[str] = None
    
    def open_for_writing(self):
        os.makedirs(self.index_dir, exist_ok=True)
        self._docs_file = open(self.docs_path, "wb")
        self._mode = "write"
        self.offsets = {}
        logger.info(f"Opened document store for writing: {self.docs_path}")
    
    def add_document(self, doc_id: str, url: str, title: str, body: str):
        """
        Write a document to disk and record its offset.
        
        doc_id: String like "D1555982" -> we extract the integer part
        """
        if self._mode != "write":
            raise RuntimeError("DocumentStore not opened for writing")
        
        # Extract integer from doc_id (e.g., "D1555982" -> 1555982)
        doc_id_int = int(doc_id[1:])
        
        # Record current position before writing
        offset = self._docs_file.tell()
        
        # Encode strings to bytes
        url_bytes = url.encode("utf-8")
        title_bytes = title.encode("utf-8")
        body_bytes = body.encode("utf-8")
        
        # Calculate total record length
        # 4 (doc_id) + 2 (url_len) + url + 2 (title_len) + title + body
        record_len = 4 + 2 + len(url_bytes) + 2 + len(title_bytes) + len(body_bytes)
        
        # Write: record_len, doc_id_int, url_len, url, title_len, title, body
        self._docs_file.write(struct.pack("<I", record_len))  # 4 bytes
        self._docs_file.write(struct.pack("<I", doc_id_int))  # 4 bytes
        self._docs_file.write(struct.pack("<H", len(url_bytes)))  # 2 bytes
        self._docs_file.write(url_bytes)
        self._docs_file.write(struct.pack("<H", len(title_bytes)))  # 2 bytes
        self._docs_file.write(title_bytes)
        self._docs_file.write(body_bytes)  # Remaining bytes are body
        
        # Store offset in memory
        self.offsets[doc_id_int] = offset
    
    def finish_writing(self):
        if self._mode != "write":
            return
        
        self._docs_file.close()
        self._write_offset_table()
        self._mode = None
        logger.info(f"Finished writing {len(self.offsets)} documents to store")
    
    def _write_offset_table(self):
        with open(self.offsets_path, "wb") as f:
            num_docs = len(self.offsets)
            max_doc_id = max(self.offsets.keys()) if self.offsets else 0
            
            # Header: num_docs, max_doc_id
            f.write(struct.pack("<II", num_docs, max_doc_id))
            
            # Write sorted (doc_id, offset) pairs
            for doc_id in sorted(self.offsets.keys()):
                offset = self.offsets[doc_id]
                f.write(struct.pack("<IQ", doc_id, offset))  # 4 + 8 = 12 bytes per entry
        
        logger.info(f"Wrote offset table with {num_docs} entries")
    
    def open_for_reading(self):
        self._load_offset_table()
        self._docs_file = open(self.docs_path, "rb")
        self._mode = "read"
        logger.info(f"Opened document store for reading: {len(self.offsets)} documents")
    
    def _load_offset_table(self):
        """Load the offset table into memory."""
        self.offsets = {}
        with open(self.offsets_path, "rb") as f:
            header = f.read(8)
            num_docs, max_doc_id = struct.unpack("<II", header)
            
            for _ in range(num_docs):
                entry = f.read(12)
                doc_id, offset = struct.unpack("<IQ", entry)
                self.offsets[doc_id] = offset
        
        logger.info(f"Loaded offset table: {num_docs} documents, max_id={max_doc_id}")
    
    def get_document(self, doc_id: str) -> Optional[dict]:
        """
        Retrieve a document by ID.
        
        Returns dict with keys: doc_id, link, title, body
        Returns None if document not found.
        """
        if self._mode != "read":
            raise RuntimeError("DocumentStore not opened for reading")
        
        # Handle both "D1555982" and just the integer
        if isinstance(doc_id, str) and doc_id.startswith("D"):
            doc_id_int = int(doc_id[1:])
        else:
            doc_id_int = int(doc_id)
        
        if doc_id_int not in self.offsets:
            return None
        
        offset = self.offsets[doc_id_int]
        self._docs_file.seek(offset)
        
        # Read record length
        record_len = struct.unpack("<I", self._docs_file.read(4))[0]
        
        # Read doc_id
        stored_doc_id = struct.unpack("<I", self._docs_file.read(4))[0]
        
        # Read url
        url_len = struct.unpack("<H", self._docs_file.read(2))[0]
        url = self._docs_file.read(url_len).decode("utf-8")
        
        # Read title
        title_len = struct.unpack("<H", self._docs_file.read(2))[0]
        title = self._docs_file.read(title_len).decode("utf-8")
        
        # Read body
        body_len = record_len - 4 - 2 - url_len - 2 - title_len
        body = self._docs_file.read(body_len).decode("utf-8")
        
        return {
            "doc_id": f"D{stored_doc_id}",
            "link": url,
            "title": title,
            "body": body
        }
    
    def close(self):
        if self._docs_file:
            self._docs_file.close()
            self._docs_file = None
        self._mode = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


"""
Disk-based Index for Query Time

This provides the same interface as the existing storage backends,
but reads posting lists directly from disk instead of keeping them in memory.

At startup:
- Loads dictionary into memory (term -> offset mapping)
- Opens postings file for seeking
- Opens document store for retrieving full documents

At query time:
- Look up term in dictionary
- Seek to posting list in postings file
- Read and return posting data
"""

import struct
import json
import os
import logging
from array import array
from typing import Optional, Generator, Any

from sea.index.document_store import DocumentStore
from sea.storage.interface import StorageInterface

logger = logging.getLogger(__name__)

# Pre-compiled struct for speed
_UINT32 = struct.Struct("<I")
_POSTING_HEADER = struct.Struct("<IHH")  # doc_id, tf, num_positions


class DiskIndex(StorageInterface):
    """
    Disk-based index that implements StorageInterface for compatibility
    with existing query operators.
    
    Usage:
        index = DiskIndex(cfg)
        index.open()
        
        # Query (same interface as LocalStorage)
        posting = index.hgetall("token:computer")
        doc = index.get("D1555982")
        
        index.close()
    """
    
    def __init__(self, cfg):
        super().__init__(cfg)
        self.index_dir = cfg.DISK_INDEX.PATH
        
        # Dictionary: term -> (offset, length) in postings file
        self.dictionary: dict[str, tuple[int, int]] = {}
        
        # File handles
        self._postings_file = None
        self._doc_store: Optional[DocumentStore] = None
        
        # State
        self._is_open = False
    
    def open(self):
        """Load dictionary and open files for reading."""
        if self._is_open:
            return
        
        logger.info(f"Opening disk index from {self.index_dir}")
        
        # Load dictionary into memory
        self._load_dictionary()
        
        # Open postings file
        postings_path = os.path.join(self.index_dir, "postings.bin")
        self._postings_file = open(postings_path, "rb")
        
        # Open document store
        self._doc_store = DocumentStore(self.index_dir)
        self._doc_store.open_for_reading()
        
        self._is_open = True
        logger.info(f"Disk index ready: {len(self.dictionary)} terms loaded")
    
    def _load_dictionary(self):
        """Load the dictionary file into memory."""
        dict_path = os.path.join(self.index_dir, "dictionary.bin")
        
        with open(dict_path, "rb") as f:
            # Read header
            num_terms = struct.unpack("<I", f.read(4))[0]
            
            # Read each entry
            for _ in range(num_terms):
                term_len = struct.unpack("<H", f.read(2))[0]
                term = f.read(term_len).decode("utf-8")
                offset = struct.unpack("<Q", f.read(8))[0]
                length = struct.unpack("<I", f.read(4))[0]
                self.dictionary[term] = (offset, length)
        
        logger.info(f"Loaded dictionary: {num_terms} terms")
    
    def hgetall(self, key: str) -> dict:
        """
        Get all postings for a token.
        
        key: "token:computer" format
        Returns: {doc_id: json_string_with_tf_and_positions, ...}
        """
        if not self._is_open:
            self.open()
        
        # Strip "token:" prefix
        term = key[6:] if key.startswith("token:") else key
        
        if term not in self.dictionary:
            return {}
        
        offset, length = self.dictionary[term]
        
        # Seek and read posting list
        self._postings_file.seek(offset)
        data = self._postings_file.read(length)
        
        # Parse posting list
        return self._parse_postings(data)
    
    def _parse_postings(self, data: bytes) -> dict:
        """Parse binary posting data into the format expected by operators."""
        result = {}
        pos = 0
        
        # Read number of documents
        num_docs = _UINT32.unpack_from(data, pos)[0]
        pos += 4
        
        for _ in range(num_docs):
            # Read header: doc_id, tf, num_positions (8 bytes total)
            doc_id, tf, num_positions = _POSTING_HEADER.unpack_from(data, pos)
            pos += 8
            
            # Read positions directly into array
            pos_array = array('I')
            pos_array.frombytes(data[pos:pos + num_positions * 4])
            pos += num_positions * 4
            
            doc_key = f"D{doc_id}"
            result[doc_key] = json.dumps({"tf": tf, "pos": pos_array.tolist()})
        
        return result
    
    def get(self, key: str) -> Optional[str]:
        """
        Get a document by ID.
        
        Returns JSON string with doc_id, link, title, body.
        """
        if not self._is_open:
            self.open()
        
        doc = self._doc_store.get_document(key)
        if doc:
            return json.dumps(doc)
        return None
    
    def hset(self, key, value):
        raise NotImplementedError("DiskIndex is read-only")
    
    def set(self, key, value):
        raise NotImplementedError("DiskIndex is read-only")
    
    def setnx(self, key, value) -> bool:
        raise NotImplementedError("DiskIndex is read-only")
    
    def delete(self, key) -> bool:
        raise NotImplementedError("DiskIndex is read-only")
    def dbsize(self) -> int:
        if self._doc_store:
            return len(self._doc_store.offsets)
        return 0
    
    def scan_iter(self, match=None, count=None) -> Generator[Any, Any, None]:
        if not self._is_open:
            self.open()
        
        yielded = 0
        for doc_id_int in self._doc_store.offsets.keys():
            doc_key = f"D{doc_id_int}"
            if match is None or doc_key.startswith(match.rstrip("*")):
                yield doc_key
                yielded += 1
                if count is not None and yielded >= count:
                    break
    
    def mget(self, keys) -> Generator[Any, Any, None]:
        """Get multiple documents by ID."""
        for key in keys:
            yield self.get(key)
    
    def pipeline(self):
        return self
    
    def execute(self) -> list:
        return [0]
    
    def ping(self) -> bool:
        return self._is_open
    
    def scan(self, cursor, count: int = 10) -> tuple:
        if not self._is_open:
            self.open()
        
        doc_ids = sorted(self._doc_store.offsets.keys())
        if cursor >= len(doc_ids):
            return 0, []
        
        next_cursor = min(cursor + count, len(doc_ids))
        keys = [f"D{doc_id}" for doc_id in doc_ids[cursor:next_cursor]]
        return next_cursor, keys
    
    def close(self):
        if self._postings_file:
            self._postings_file.close()
            self._postings_file = None
        if self._doc_store:
            self._doc_store.close()
            self._doc_store = None
        self._is_open = False
        self.dictionary.clear()
        logger.info("Disk index closed")
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


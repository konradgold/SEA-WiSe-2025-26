"""
SPIMI Implementation 

This builds an inverted index while respecting memory limits:
1. Accumulate postings in memory until we hit the memory threshold
2. Sort and write a "block" to disk
3. After processing all documents, merge all blocks into the final index

Optimizations:
- Batched binary I/O using struct.pack with format strings
- array module for position lists (direct buffer write)
- Buffered file I/O
"""

import struct
import os
import io
import heapq
import logging
from array import array
from collections import defaultdict

logger = logging.getLogger(__name__)

# Pre-compiled struct formats for speed
_UINT16 = struct.Struct("<H")
_UINT32 = struct.Struct("<I")
_UINT64 = struct.Struct("<Q")
_POSTING_HEADER = struct.Struct("<IHH")  # doc_id, tf, num_positions


class SPIMIIndexer:
    """
    Builds an inverted index using SPIMI strategy.
    
    Optimized for speed with batched I/O operations.
    """
    
    def __init__(self, index_dir: str, memory_limit_mb: int = 800):
        self.index_dir = index_dir
        self.blocks_dir = os.path.join(index_dir, "blocks")
        self.memory_limit = memory_limit_mb * 1024 * 1024
        
        # In-memory postings: term -> list of (doc_id_int, tf, positions_array)
        self.postings: dict[str, list[tuple[int, int, array]]] = defaultdict(list)
        
        self.current_memory = 0
        self.block_count = 0
        self.total_docs = 0
        
        os.makedirs(self.blocks_dir, exist_ok=True)
    
    def add_document(self, doc_id: str, tokens: list[str]):
        """Add a document's tokens to the index."""
        if not tokens:
            return
        
        doc_id_int = int(doc_id[1:])
        
        # Build term frequencies and positions
        term_positions: dict[str, list[int]] = defaultdict(list)
        for pos, token in enumerate(tokens):
            if token:
                term_positions[token].append(pos)
        
        # Add to postings with array for positions 
        for term, positions in term_positions.items():
            pos_array = array('I', positions)  # unsigned int array
            tf = len(positions)
            self.postings[term].append((doc_id_int, tf, pos_array))
            
            # Memory estimate: ~40 bytes base + 4 bytes per position
            self.current_memory += 40 + len(positions) * 4
        
        self.total_docs += 1
        
        if self.current_memory >= self.memory_limit:
            self._flush_block()
    
    def _flush_block(self):
        """Sort and write postings to a block file."""
        if not self.postings:
            return
        
        block_path = os.path.join(self.blocks_dir, f"block_{self.block_count:04d}.bin")
        logger.info(f"Flushing block {self.block_count} with {len(self.postings)} terms "
                   f"(~{self.current_memory / 1024 / 1024:.1f} MB)")
        
        # Use buffered writer for better I/O performance
        with io.BufferedWriter(io.FileIO(block_path, "wb"), buffer_size=8*1024*1024) as f:
            # Sort terms and write
            for term in sorted(self.postings.keys()):
                postings_list = self.postings[term]
                postings_list.sort(key=lambda x: x[0])  # Sort by doc_id
                self._write_term_postings_fast(f, term, postings_list)
        
        self.postings.clear()
        self.current_memory = 0
        self.block_count += 1
    
    def _write_term_postings_fast(self, f, term: str, postings_list: list):
        """Write term postings with optimized binary I/O."""
        term_bytes = term.encode("utf-8")
        
        # Write term header
        f.write(_UINT16.pack(len(term_bytes)))
        f.write(term_bytes)
        f.write(_UINT32.pack(len(postings_list)))
        
        # Write all postings
        for doc_id, tf, pos_array in postings_list:
            f.write(_POSTING_HEADER.pack(doc_id, tf, len(pos_array)))
            # Write positions directly from array buffer
            f.write(pos_array.tobytes())
    
    def finalize(self):
        """Flush remaining postings and merge all blocks."""
        self._flush_block()
        
        logger.info(f"Merging {self.block_count} blocks into final index...")
        
        if self.block_count == 0:
            logger.warning("No blocks to merge")
            return
        
        self._merge_blocks()
        self._cleanup_blocks()
        
        logger.info(f"Indexing complete: {self.total_docs} documents indexed")
    
    def _merge_blocks(self):
        """K-way merge with optimized I/O."""
        dict_path = os.path.join(self.index_dir, "dictionary.bin")
        postings_path = os.path.join(self.index_dir, "postings.bin")
        
        # Open block readers
        block_readers = []
        for i in range(self.block_count):
            block_path = os.path.join(self.blocks_dir, f"block_{i:04d}.bin")
            reader = BlockReader(block_path)
            block_readers.append(reader)
        
        # Initialize heap
        heap = []
        for i, reader in enumerate(block_readers):
            entry = reader.read_next()
            if entry:
                term, postings = entry
                heapq.heappush(heap, (term, i, postings))
        
        # Merge and write
        dictionary_entries = []
        
        with io.BufferedWriter(io.FileIO(postings_path, "wb"), buffer_size=8*1024*1024) as postings_file:
            current_term = None
            current_postings = []
            
            while heap:
                term, block_idx, postings = heapq.heappop(heap)
                
                if current_term is not None and term != current_term:
                    offset, length = self._write_merged_postings_fast(
                        postings_file, current_postings
                    )
                    dictionary_entries.append((current_term, offset, length))
                    current_postings = []
                
                current_term = term
                current_postings.extend(postings)
                
                entry = block_readers[block_idx].read_next()
                if entry:
                    next_term, next_postings = entry
                    heapq.heappush(heap, (next_term, block_idx, next_postings))
            
            # Write last term
            if current_term is not None and current_postings:
                offset, length = self._write_merged_postings_fast(
                    postings_file, current_postings
                )
                dictionary_entries.append((current_term, offset, length))
        
        for reader in block_readers:
            reader.close()
        
        self._write_dictionary_fast(dict_path, dictionary_entries)
        logger.info(f"Merged index: {len(dictionary_entries)} unique terms")
    
    def _write_merged_postings_fast(self, f, postings: list) -> tuple[int, int]:
        """Write merged postings with optimized I/O."""
        offset = f.tell()
        
        # Sort by doc_id
        postings.sort(key=lambda x: x[0])
        
        # Merge duplicates
        merged = []
        for doc_id, tf, pos_array in postings:
            if merged and merged[-1][0] == doc_id:
                old_id, old_tf, old_pos = merged[-1]
                combined = array('I', old_pos)
                combined.extend(pos_array)
                merged[-1] = (doc_id, old_tf + tf, combined)
            else:
                merged.append((doc_id, tf, pos_array))
        
        # Write count
        f.write(_UINT32.pack(len(merged)))
        
        # Write all postings
        for doc_id, tf, pos_array in merged:
            f.write(_POSTING_HEADER.pack(doc_id, tf, len(pos_array)))
            f.write(pos_array.tobytes())
        
        return offset, f.tell() - offset
    
    def _write_dictionary_fast(self, path: str, entries: list):
        """Write dictionary with optimized I/O."""
        with io.BufferedWriter(io.FileIO(path, "wb"), buffer_size=4*1024*1024) as f:
            f.write(_UINT32.pack(len(entries)))
            
            for term, offset, length in entries:
                term_bytes = term.encode("utf-8")
                f.write(_UINT16.pack(len(term_bytes)))
                f.write(term_bytes)
                f.write(_UINT64.pack(offset))
                f.write(_UINT32.pack(length))
        
        logger.info(f"Wrote dictionary with {len(entries)} terms")
    
    def _cleanup_blocks(self):
        """Remove temporary block files."""
        for i in range(self.block_count):
            block_path = os.path.join(self.blocks_dir, f"block_{i:04d}.bin")
            try:
                os.remove(block_path)
            except OSError:
                pass
        try:
            os.rmdir(self.blocks_dir)
        except OSError:
            pass


class BlockReader:
    """Optimized block reader with buffered I/O."""
    
    def __init__(self, path: str):
        self.file = io.BufferedReader(io.FileIO(path, "rb"), buffer_size=4*1024*1024)
    
    def read_next(self) -> tuple[str, list] | None:
        """Read next term and its postings."""
        data = self.file.read(2)
        if not data:
            return None
        
        term_len = _UINT16.unpack(data)[0]
        term = self.file.read(term_len).decode("utf-8")
        
        num_postings = _UINT32.unpack(self.file.read(4))[0]
        
        postings = []
        for _ in range(num_postings):
            header = self.file.read(8)
            doc_id, tf, num_pos = _POSTING_HEADER.unpack(header)
            
            # Read positions directly into array
            pos_bytes = self.file.read(num_pos * 4)
            pos_array = array('I')
            pos_array.frombytes(pos_bytes)
            
            postings.append((doc_id, tf, pos_array))
        
        return term, postings
    
    def close(self):
        self.file.close()

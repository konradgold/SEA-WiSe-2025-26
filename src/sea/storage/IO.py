from array import array
import collections
import os
import struct
from typing import Dict, Tuple
from sea.utils.config import Config


class BlockIO:
    def __init__(self):
        cfg = Config(load=True)
        self.magic_header_binary = cfg.HEADER_BLOCK_FILE.encode("utf-8")
        self.block_path = cfg.BLOCK_PATH
    
    def write_block(self, block_id: str, index: Dict[str, array]):
        path = self._get_block_path(block_id)
        ordered_index = collections.OrderedDict(sorted(index.items()))

        with open(path, "wb") as out:
            out.write(self.magic_header_binary) 
            for term, arr in ordered_index.items():
                # ensure uint32 array
                if arr.typecode != "I":
                    arr = array("I", arr)

                tb = term.encode("utf-8")
                out.write(struct.pack("<I", len(tb)))
                out.write(tb)

                out.write(struct.pack("<I", len(arr)))
                out.write(arr.tobytes())

    def _get_block_path(self, block_id: str) -> str:
        return os.path.join(self.block_path, f"tokenizer_output_{block_id}.bin")

    def check_magic_header(self, file) -> None:
        magic_version = file.read(5)
        if magic_version != self.magic_header_binary:
            raise ValueError("Bad magic/version")
        
    def read_line(self, file) -> Tuple[str, array]:
        lb = file.read(4)
        if not lb:
            return None, None
        (term_len,) = struct.unpack("<I", lb)
        term = file.read(term_len).decode("utf-8")
        (count,) = struct.unpack("<I", file.read(4))
        buf = file.read(count * 4)
        arr = array("I")
        arr.frombytes(buf)  # uint32 little-endian
        return term, arr
    
class IndexIO():
    def __init__(self, rewrite: bool = False):
        self.rewrite = rewrite
        self.index_file, self.posting_file = self._open_files(rewrite)

    def _open_files(self, rewrite: bool):
        cfg = Config(load=True)

        index_path = os.path.join(cfg.DATA_PATH, "term_dictionary.bin")
        posting_path = os.path.join(cfg.DATA_PATH, "posting_list.bin")

        header_index_binary = cfg.HEADER_INDEX_FILE.encode("utf-8")
        header_posting_binary = cfg.HEADER_POSTING_FILE.encode("utf-8")
        if rewrite:
             index_file = open(index_path, "w+b")
             posting_file = open(posting_path, "w+b")
             index_file.write(header_index_binary)
             posting_file.write(header_posting_binary)
        else:
            index_file = open(index_path, "rb")
            self._check_magic_header(index_file, header_index_binary)
            posting_file = open(posting_path, "rb")
            self._check_magic_header(posting_file, header_posting_binary)
        return index_file, posting_file

    def _check_magic_header(self, file, expected_header: bytes):
        magic_version = file.read(len(expected_header))
        if magic_version != expected_header:
            raise ValueError("Bad magic/version")
        
    def write_line(self, term: str, posting_list: array):
        if self.rewrite:
            disk_offset, length = self._write_posting_entry(posting_list)
            self._write_index_entry(term, disk_offset, length)
        else:
            raise RuntimeError("IndexIO opened in read-only mode (rewrite=False); cannot write. Set rewrite=True to enable writing.")

    def _write_posting_entry(self, posting_list: array) -> Tuple[int, int]:
        start = self.posting_file.tell()

        self.posting_file.write(struct.pack("<I", len(posting_list)))
        self.posting_file.write(posting_list.tobytes())

        end = self.posting_file.tell()
        return [start, end - start]
        
    def _write_index_entry(self, term: str, disk_offset: int, posting_length: int) -> None:
        term = term.encode("utf-8")
        self.index_file.write(struct.pack("<I", len(term)))
        self.index_file.write(term)

        self.index_file.write(struct.pack("<I", disk_offset))      
        self.index_file.write(struct.pack("<I", posting_length))

    def close(self):
        self.index_file.close()
        self.posting_file.close()

from array import array
import collections
import os
import struct
from typing import Dict, Optional, Tuple
from sea.utils.config import Config

class BlockIO:
    # BE aware of magic header when reading/writing!
    #
    # structure [uint32][bytes][uint32][uint32]* -> [7][122][1][56][123][2][34][45]
    # [uint32] length of term in bytes
    # [bytes] term encoded in utf-8
    # [uint32] count of uint32 values in this posting list
    # [uint32]* contiguous uint32 values (rythme: docID, term frequency, positions, docID...)
    #
    # Example on disk after header: [7][122][1][56][123][2][34][45]
    #  - 7 = number of uint32 values
    #  - 122 = internal doc ID
    #  - 1 = term frequency in document
    #  - 56 = position 1
    #  - 123 = internal doc ID
    #  - 2 = term frequency in document
    #  - 34 = position 1
    #  - 45 = position 2
    #
    # BE aware of magic header when reading/writing!
    def __init__(self, cfg: Optional[Config] = None):
        if cfg is None:
            cfg = Config(load=True)
        self.magic_header_binary = cfg.HEADER_BLOCK_FILE.encode("utf-8")
        self.block_path = cfg.BLOCK_PATH
        self.store_positions = cfg.TOKENIZER.STORE_POSITIONS
    
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
        
    def read_line(self, file) -> Optional[Tuple[str, array]]:
    # if not self.store_positions array includes only the term frequency
        lb = file.read(4)
        if not lb:
            return None, None
        (term_len,) = struct.unpack("<I", lb)
        term = file.read(term_len).decode("utf-8")

        cb = file.read(4)
        if len(cb) != 4:
            raise EOFError(f"{getattr(file, 'name','<file>')}: truncated count")
        (count,) = struct.unpack("<I", cb)

        payload = file.read(count * 4)
        if len(payload) != count * 4:
            raise EOFError(f"{getattr(file, 'name','<file>')}: truncated payload")
        arr = array("I")
        arr.frombytes(payload)
        return term, arr

class TermDictionaryIO():
    # BE aware of magic header when reading/writing!
    #
    # structure [uint32][bytes][uint64][uint64] -> [5][b'apple'][12345][678]
    # [uint32] length of term in bytes
    # [bytes] term encoded in utf-8
    # [uint64] disk offset of posting list in posting_list.bin
    # [uint64] length of posting list in bytes
    #
    # BE aware of magic header when reading/writing!
    def __init__(self, rewrite: bool = False, cfg: Optional[Config] = None):
        self.rewrite = rewrite
        if cfg is None:
            cfg = Config(load=True)
        self.index_file = self._open_file(rewrite, cfg)

    def _open_file(self, rewrite: bool, cfg: Config):

        index_path = os.path.join(cfg.DATA_PATH, "term_dictionary.bin")

        header_index_binary = cfg.HEADER_INDEX_FILE.encode("utf-8")
        if rewrite:
            index_file = open(index_path, "w+b")
            index_file.write(header_index_binary)
        else:
            index_file = open(index_path, "rb")
            self._check_magic_header(index_file, header_index_binary)
        return index_file

    def _check_magic_header(self, file, expected_header: bytes):
        magic_version = file.read(len(expected_header))
        if magic_version != expected_header:
            raise ValueError("Bad magic/version")

    def write(self, term: str, disk_offset: int, posting_length: int) -> None:
        term = term.encode("utf-8")
        self.index_file.write(struct.pack("<I", len(term)))
        self.index_file.write(term)

        self.index_file.write(struct.pack("<Q", disk_offset))
        self.index_file.write(struct.pack("<Q", posting_length))

    def read(self) -> Dict[str, Tuple[int, int]]:
        term_dict: Dict[str, Tuple[int, int]] = {}
        while True:
            term, (disk_offset, length) = self.read_line()
            if not term:
                break
            term_dict[term] = (disk_offset, length)
        return term_dict

    def read_line(self) -> Tuple[str, Tuple[int, int]]:
        lb = self.index_file.read(4)
        if not lb:
            return None, (0, 0)
        (term_len,) = struct.unpack("<I", lb)
        term = self.index_file.read(term_len).decode("utf-8")
        (disk_offset,) = struct.unpack("<Q", self.index_file.read(8))
        (length,) = struct.unpack("<Q", self.index_file.read(8))
        return term, (disk_offset, length)

    def close(self):
        self.index_file.close()

class PostingListIO():
    # BE aware of magic header when reading/writing!
    #
    # structure [uint32][uint32]* -> [7][122][1][56][123][2][34][45]
    # [uint32] count of uint32 values in this posting list
    # [uint32]* contiguous uint32 values  (rythme: docID, term frequency, positions, docID...)
    #
    # Example on disk after header: [7][122][1][56][123][2][34][45]
    #  - 7 = number of uint32 values
    #  - 122 = internal doc ID
    #  - 1 = term frequency in document
    #  - 56 = position 1
    #  - 123 = internal doc ID
    #  - 2 = term frequency in document
    #  - 34 = position 1
    #  - 45 = position 2
    #
    # BE aware of magic header when reading/writing!
    def __init__(self, rewrite: bool = False, cfg: Optional[Config] = None):
            self.rewrite = rewrite
            if cfg is None:
                cfg = Config(load=True)
            self.posting_file = self._open_file(rewrite, cfg)

    def _open_file(self, rewrite: bool, cfg: Config):

        posting_path = os.path.join(cfg.DATA_PATH, "posting_list.bin")

        header_posting_binary = cfg.HEADER_POSTING_FILE.encode("utf-8")
        if rewrite:
             posting_file = open(posting_path, "w+b")
             posting_file.write(header_posting_binary)
        else:
            posting_file = open(posting_path, "rb")
            self._check_magic_header(posting_file, header_posting_binary)
        return posting_file

    def _check_magic_header(self, file, expected_header: bytes):
        magic_version = file.read(len(expected_header))
        if magic_version != expected_header:
            raise ValueError("Bad magic/version")
        
    def read(self, disk_offset: int, length: int) -> array:
        self.posting_file.seek(disk_offset)
        buf = self.posting_file.read(length)
        arr = array("I")
        arr.frombytes(buf)  # uint32 little-endian
        return arr
        
    # returns (disk_offset, length)
    def write(self, posting_list: array) -> Tuple[int, int]:
        start = self.posting_file.tell()

        self.posting_file.write(struct.pack("<I", len(posting_list)))
        self.posting_file.write(posting_list.tobytes())

        end = self.posting_file.tell()
        return (start, end - start)
        
    def close(self):
        self.posting_file.close()


class DocDictonaryIO():
    # BE aware of magic header when reading/writing!
    #
    # structure [uint32][uint32][bytes][uint32] -> [123][5][b'D2919'][42]
    # [uint32] document id (internal integer id)
    # [uint32] length of original document id in bytes
    # [bytes] original document id encoded in utf-8
    # [uint32] token count (number of tokens in the document)
    #
    # BE aware of magic header when reading/writing!
    def __init__(self, rewrite: bool = False, cfg: Optional[Config] = None):
        self.rewrite = rewrite
        if cfg is None:
            cfg = Config(load=True)
        self.doc_dict_file = self._open_file(rewrite, cfg)

    def _open_file(self, rewrite: bool, cfg: Config):

        doc_dict_path = os.path.join(cfg.DATA_PATH, "doc_dictionary.bin")
        header_doc_dict_binary = cfg.HEADER_DOC_DICT_FILE.encode("utf-8")
        if rewrite:
            doc_dict_file = open(doc_dict_path, "w+b")
            doc_dict_file.write(header_doc_dict_binary)
        else:
            doc_dict_file = open(doc_dict_path, "rb")
            self._check_magic_header(doc_dict_file, header_doc_dict_binary)
        return doc_dict_file

    def _check_magic_header(self, file, expected_header: bytes):
        magic_version = file.read(len(expected_header))
        if magic_version != expected_header:
            raise ValueError("Bad magic/version")

    def write_metadata(self, metadata: Dict[int, Tuple[str, int]]):
        import tqdm

        if self.rewrite:
            print(
                f"Writing {len(metadata):,} metadata entries to {self.doc_dict_file.name}..."
            )
            for doc_id, meta in tqdm.tqdm(metadata.items(), desc="Writing DocDict"):
                original_id = meta[0].encode("utf-8")
                token_count = meta[1]
                self.doc_dict_file.write(struct.pack("<I", doc_id))
                self.doc_dict_file.write(struct.pack("<I", len(original_id)))
                self.doc_dict_file.write(original_id)
                self.doc_dict_file.write(struct.pack("<I", token_count))
            self.doc_dict_file.flush()
        else:
            raise RuntimeError("DocDictonaryIO opened in read-only mode (rewrite=False); cannot write. Set rewrite=True to enable writing.")

    def read(self) -> Dict[int, Tuple[str, int]]:
        metadata: Dict[int, Tuple[str, int]] = {}
        while True:
            lb = self.doc_dict_file.read(4)
            if not lb:
                break
            (doc_id,) = struct.unpack("<I", lb)
            (original_id_len,) = struct.unpack("<I", self.doc_dict_file.read(4))
            original_id = self.doc_dict_file.read(original_id_len).decode("utf-8")
            (token_count,) = struct.unpack("<I", self.doc_dict_file.read(4))
            metadata[doc_id] = (original_id, token_count)
        return metadata

    def close(self):
        self.doc_dict_file.close()

from array import array
from typing import Dict, Optional, Tuple

from sea.storage.IO import DocDictonaryIO, TermDictionaryIO, PostingListIO
from sea.utils.config import Config


class StorageManager:

    # if rewrite is True, files are opened in write mode (existing files will be overwritten)
    # if rewrite is False, files are opened in read mode
    def __init__(self, rewrite: bool = False, cfg: Optional[Config] = None):
        self.rewrite = rewrite
        if cfg is None:
            cfg = Config(load=True)
        self.termDictionaryIO = TermDictionaryIO(rewrite=rewrite, cfg=cfg)
        self.postingListIO = PostingListIO(rewrite=rewrite, cfg=cfg)
        self.DocDictionaryIO = DocDictonaryIO(rewrite=rewrite, cfg=cfg)

        self.termDictionary: Dict[str, Tuple[int, int]] = {}
        self.docMetadata: Dict[int, Tuple[str, int]] = {}

    def init_all(self):
        self.getDocMetadata()

    def write_term_posting_list(self, term: str, posting_list: array):
        if self.rewrite:
            disk_offset, length = self.postingListIO.write(posting_list)
            self.termDictionaryIO.write(term, disk_offset, length)
        else:
            raise RuntimeError("IndexIO opened in read-only mode (rewrite=False); cannot write. Set rewrite=True to enable writing.")

    def getTermDictionary(self) -> Dict[str, Tuple[int, int]]:
        if not self.termDictionary:
            self.termDictionary = self.termDictionaryIO.read()
        return self.termDictionary
    
    def getDocMetadata(self) -> Dict[int, Tuple[str, int]]:
        if not self.docMetadata:
            self.docMetadata = self.DocDictionaryIO.read()
        return self.docMetadata

    def getPostingList(self, term :str) -> array | None:
        disk_offset, length = self.getTermDictionary().get(term, (-1, -1))
        if disk_offset == -1:
            return array("I")  # empty posting list
        return self.postingListIO.read(disk_offset, length)
    
    def getDocMetadataEntry(self, doc_id: int) -> Tuple[str, int]:
        return self.getDocMetadata().get(doc_id, ("", 0))

    def close(self):
        self.termDictionaryIO.close()
        self.postingListIO.close()
        self.DocDictionaryIO.close()
from array import array
from typing import Dict, Optional, Tuple

from sea.storage.IO import DocDictionaryIO, TermDictionaryIO, PostingListIO
from sea.utils.config_wrapper import Config
from omegaconf import DictConfig


class StorageManager:
    """I/O manager for the search engine's disk-based index.

    Coordinates access to three key data structures:
    - Term Dictionary: Maps terms to posting list locations on disk
    - Posting Lists: Contains document IDs and term frequencies
    - Document Dictionary: Maps internal IDs to original IDs and metadata

    Args:
        rewrite: If True, opens files in write mode (overwrites existing).
        cfg: OmegaConf configuration object.
        rewrite_doc_dict: Override for doc dictionary rewrite behavior.
        field: Optional field name for fielded search indexes.
    """

    def __init__(
        self,
        rewrite: bool = False,
        cfg: Optional[DictConfig] = None,
        rewrite_doc_dict: Optional[bool] = None,
        field: Optional[str] = None
    ):
        self.rewrite = rewrite
        if cfg is None:
            cfg = Config(load=True)
        self.termDictionaryIO = TermDictionaryIO(rewrite=rewrite, cfg=cfg, field=field)
        self.postingListIO = PostingListIO(rewrite=rewrite, cfg=cfg, field=field)

        # Only rewrite doc dict if explicitly requested or if rewrite is True and not explicitly disabled
        do_rewrite_doc = rewrite if rewrite_doc_dict is None else rewrite_doc_dict
        self.DocDictionaryIO = DocDictionaryIO(rewrite=do_rewrite_doc, cfg=cfg, field=field)

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

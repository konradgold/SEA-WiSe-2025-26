"""Binary storage for document embeddings, aligned with existing index format."""

import mmap
import os
import struct
from typing import Optional

import numpy as np
from omegaconf import DictConfig

from sea.utils.config_wrapper import Config

# Magic header: "SEAV" + version byte (V for vectors)
HEADER_EMBEDDING_FILE = b"SEAV\x01"


class EmbeddingIO:
    """
    Binary format for embeddings:
    [5 bytes] magic header "SEAV\x01"
    [uint32]  num_docs
    [uint32]  dim
    [float32 * num_docs * dim] embedding data (row-major)

    Supports memory-mapped reading for large files.
    """

    def __init__(self, cfg: Optional[DictConfig] = None):
        if cfg is None:
            cfg = Config(load=True)
        self.cfg = cfg
        self.data_path = cfg.DATA_PATH
        self._mmap: Optional[mmap.mmap] = None
        self._file = None
        self._num_docs = 0
        self._dim = 0
        self._data_offset = 0

    def _get_path(self) -> str:
        return os.path.join(self.data_path, "embeddings.bin")

    def write(self, embeddings: np.ndarray) -> None:
        """Write embeddings to binary file."""
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        num_docs, dim = embeddings.shape
        path = self._get_path()

        with open(path, "wb") as f:
            f.write(HEADER_EMBEDDING_FILE)
            f.write(struct.pack("<I", num_docs))
            f.write(struct.pack("<I", dim))
            f.write(embeddings.tobytes())

        print(f"Wrote {num_docs:,} embeddings ({dim} dims) to {path}")

    def open_readonly(self) -> None:
        """Open file for memory-mapped reading."""
        path = self._get_path()
        self._file = open(path, "rb")

        # Check header
        header = self._file.read(5)
        if header != HEADER_EMBEDDING_FILE:
            raise ValueError(f"Invalid embedding file header: {header}")

        self._num_docs = struct.unpack("<I", self._file.read(4))[0]
        self._dim = struct.unpack("<I", self._file.read(4))[0]
        self._data_offset = self._file.tell()

        # Memory-map the file for efficient access
        self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)

    def load_all(self) -> np.ndarray:
        """Load all embeddings into memory as numpy array."""
        if self._mmap is None:
            self.open_readonly()

        data_size = self._num_docs * self._dim * 4  # float32 = 4 bytes
        self._mmap.seek(self._data_offset)
        data = self._mmap.read(data_size)
        return np.frombuffer(data, dtype=np.float32).reshape(self._num_docs, self._dim)

    def get_embedding(self, doc_id: int) -> np.ndarray:
        """Get single embedding by document ID (row index)."""
        if self._mmap is None:
            self.open_readonly()

        offset = self._data_offset + doc_id * self._dim * 4
        self._mmap.seek(offset)
        data = self._mmap.read(self._dim * 4)
        return np.frombuffer(data, dtype=np.float32)

    @property
    def num_docs(self) -> int:
        if self._mmap is None:
            self.open_readonly()
        return self._num_docs

    @property
    def dim(self) -> int:
        if self._mmap is None:
            self.open_readonly()
        return self._dim

    def close(self) -> None:
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
        if self._file is not None:
            self._file.close()
            self._file = None

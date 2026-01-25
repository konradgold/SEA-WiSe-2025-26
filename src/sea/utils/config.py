from dataclasses import dataclass, field
from typing import List
from hydra.core.config_store import ConfigStore

@dataclass
class TokenizerConfig:
    MIN_LEN: int = 2
    BACKEND: str = "simple"
    LOWERCASE: bool = True
    ASCII_FOLD: bool = False
    REMOVE_STOPWORDS: bool = True
    STEM: bool = False
    NUMBER_NORMALIZE: bool = False
    NUM_WORKERS: int = 0
    STORE_POSITIONS: bool = False
    STORE_TOKENS: bool = False


@dataclass
class ChunkerConfig:
    MAX_CHUNK_SIZE: int = 80
    MIN_CHUNK_SIZE: int = 30
    CHUNK_OVERLAP: int = 10
    ENABLE: bool = True


@dataclass
class FIELDEDConfig:
    ACTIVE: bool = True
    FIELDS: List[str] = field(default_factory=lambda: ["title", "body", "url"])
    LENGTHS: dict = field(default_factory=lambda: {
        "title": 20.0,
        "body": 80.0,
        "url": 10.0
    })
    WEIGHTS: dict = field(default_factory=lambda: {
        "title": 2.0,
        "body": 1.0,
        "url": 1.5
    })


@dataclass
class RerankerConfig:
    ENABLED: bool = False
    MODEL_PATH: str | None = None
    CANDIDATE_TOPN: int = 200


@dataclass
class SemanticConfig:
    SERVICE_URL: str = "http://localhost:8001"
    SERVICE_PORT: int = 8001
    MODEL_ID: str = "nomic-ai/nomic-embed-text-v1.5"
    DIM: int = 64
    DEVICE: str = "cpu"
    BATCH_SIZE: int = 64


@dataclass
class SearchConfig:
    MAX_RESULTS: int = 10
    POSTINGS_CUT: int = 100000
    EXPAND_QUERIES: bool = False
    RETRIEVAL: str = "bm25"
    NUM_DOCS: int = 100000
    AVG_DOC_LEN: float = 100.0
    VERBOSE_OUTPUT: bool = False
    FIELDED: FIELDEDConfig = field(default_factory=FIELDEDConfig)
    RERANKER: RerankerConfig = field(default_factory=RerankerConfig)

@dataclass
class BM25Config:
    K1: float = 1.5
    B: float = 0.75

@dataclass
class IngestionConfig:
    NUM_DOCUMENTS: int = -1
    BATCH_SIZE: int = 1000

@dataclass
class QueryConfig:
    MAX_PHRASE_LEN: int = 5

@dataclass
class SpladeConfig:
    MODEL_ID: str = "naver/splade-cocondenser-ensembledistil"
    DEVICE: str = "cuda:0"
    THRESHOLD: float = 0.01
    CACHE_DIR: str = "cache/splade"
    CAP_EXPANSION: int = 2

@dataclass
class SchedulerConfig:
    TYPE: str = "cosine"
    DECAY_STEPS: int = 10000
    DECAY_RATE: float = 0.9
    ALPHA: float = 0.1

@dataclass
class LTRModelConfig:
    HIDDEN_UNITS: List[int] = field(default_factory=lambda: [512, 256, 128])
    DROPOUT: float = 0.1
    USE_ATTENTION: bool = True

@dataclass
class WandBConfig:
    PROJECT: str = "SEA-WiSe-2025-26"
    GROUP: str = "reranker"
    LOG_MODEL: bool = True

@dataclass
class LTRConfig:
    DATA_DIR: str = "data"
    QUERIES: str = "data/msmarco-doctrain-queries.tsv"
    QRELS: str = "data/msmarco-doctrain-qrels.tsv"
    SPLIT_DIR: str = "data/splits"
    TRAIN_CACHE: str = "data/train_cache_20k.npz"
    VAL_CACHE: str = "data/val_cache_5k.npz"
    CANDIDATE_TOPN: int = 200
    LIST_SIZE: int = 100
    EPOCHS: int = 1
    BATCH_SIZE: int = 1024
    LEARNING_RATE: float = 0.001
    SCHEDULER: SchedulerConfig = field(default_factory=SchedulerConfig)
    SEED: int = 42
    MODEL: LTRModelConfig = field(default_factory=LTRModelConfig)
    WANDB: WandBConfig = field(default_factory=WandBConfig)
    FEATURES: List[str] = field(default_factory=lambda: [
        "bm25_score", "query_len", "query_uniq_len", "title_len", "body_len",
        "title_overlap_cnt", "body_overlap_cnt", "body_overlap_ratio",
        "idf_body_overlap_sum", "idf_title_overlap_sum"
    ])

@dataclass
class MainConfig:
    TOKENIZER: TokenizerConfig = field(default_factory=TokenizerConfig)
    CHUNKER: ChunkerConfig = field(default_factory=ChunkerConfig)
    SEARCH: SearchConfig = field(default_factory=SearchConfig)
    BM25: BM25Config = field(default_factory=BM25Config)
    INGESTION: IngestionConfig = field(default_factory=IngestionConfig)
    QUERY: QueryConfig = field(default_factory=QueryConfig)
    SPLADE: SpladeConfig = field(default_factory=SpladeConfig)
    LTR: LTRConfig = field(default_factory=LTRConfig)
    SEMANTIC: SemanticConfig = field(default_factory=SemanticConfig)
    
    DOCUMENTS: str = "data/msmarco-docs.tsv"
    DOCUMENT_OFFSETS: str = "data/msmarco-offsets.pkl"
    INDEX_INTERVAL: int = 50
    DATA_PATH: str = "data/"
    BLOCK_PATH: str = "data/blocks/"
    HEADER_BLOCK_FILE: str = "SEAB\x01"
    HEADER_INDEX_FILE: str = "SEAI\x01"
    HEADER_POSTING_FILE: str = "SEAP\x01"
    HEADER_DOC_DICT_FILE: str = "SEAD\x01"
    LOG_PATH: str = "data/log.txt"

def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=MainConfig)

# Automatically register when this module is imported

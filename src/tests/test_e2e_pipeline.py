"""
End-to-end integration tests for the full search pipeline.

These tests use minimal real data and process actual files, without
overwriting the existing production index files.
"""

import os
import shutil
import tempfile
import unittest
from array import array
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np

from omegaconf import OmegaConf


# =====================================================================
# TEST FIXTURES - Minimal realistic data
# =====================================================================

# 5 minimal documents in TSV format (doc_id, url, title, body)
SAMPLE_DOCS_TSV = """\
D001\thttp://example.com/apple\tApple Fruit Guide\tApples are delicious fruits. They come in red and green varieties. Apple pie is a popular dessert.
D002\thttp://example.com/banana\tBanana Information\tBananas are yellow tropical fruits. They are rich in potassium and vitamins.
D003\thttp://example.com/recipe\tFruit Recipe Collection\tMake apple pie with fresh apples. Banana smoothie is healthy. Fruit salad recipe included.
D004\thttp://example.com/health\tHealthy Eating Guide\tEat fruits daily for good health. Apples and bananas are nutritious choices.
D005\thttp://example.com/garden\tGardening Tips\tGrow apple trees in your garden. Plant banana plants in tropical climates.
"""

# Sample queries (qid, query_text)
SAMPLE_QUERIES_TSV = """\
Q001\tapple pie recipe
Q002\tbanana health benefits
Q003\tfruit gardening
"""

# Sample qrels (qid, unused, doc_id, relevance)
SAMPLE_QRELS_TSV = """\
Q001\t0\tD001\t1
Q001\t0\tD003\t1
Q002\t0\tD002\t1
Q002\t0\tD004\t1
Q003\t0\tD005\t1
"""


def create_test_config(temp_dir: str, fields: list[str] | None = None) -> OmegaConf:
    """Create a test configuration pointing to temporary directory."""
    if fields is None:
        fields = ["title", "body"]
    config_dict = {
        "TOKENIZER": {
            "MIN_LEN": 2,
            "BACKEND": "simple",
            "LOWERCASE": True,
            "ASCII_FOLD": False,
            "REMOVE_STOPWORDS": True,
            "STEM": False,
            "NUMBER_NORMALIZE": False,
            "NUM_WORKERS": 1,  # Single worker for tests
            "STORE_POSITIONS": False,
            "STORE_TOKENS": False,
        },
        "CHUNKER": {
            "MAX_CHUNK_SIZE": 80,
            "MIN_CHUNK_SIZE": 30,
            "CHUNK_OVERLAP": 10,
            "ENABLE": True,
        },
        "DOCUMENTS": os.path.join(temp_dir, "docs.tsv"),
        "DOCUMENT_OFFSETS": os.path.join(temp_dir, "offsets.pkl"),
        "INDEX_INTERVAL": 2,  # Small interval for test
        "DATA_PATH": temp_dir,
        "BLOCK_PATH": os.path.join(temp_dir, "blocks/"),
        "HEADER_BLOCK_FILE": "SEAB\x01",
        "HEADER_INDEX_FILE": "SEAI\x01",
        "HEADER_POSTING_FILE": "SEAP\x01",
        "HEADER_DOC_DICT_FILE": "SEAD\x01",
        "LOG_PATH": os.path.join(temp_dir, "log.txt"),
        "SEARCH": {
            "RETRIEVAL": "bm25",
            "MAX_RESULTS": 5,
            "POSTINGS_CUT": 100,
            "EXPAND_QUERIES": False,
            "NUM_DOCS": 5,
            "AVG_DOC_LEN": 20.0,
            "VERBOSE_OUTPUT": False,
            "RERANKER": {
                "ENABLED": False,
                "MODEL_PATH": os.path.join(temp_dir, "model.keras"),
                "CANDIDATE_TOPN": 5,
            },
            "FIELDED": {
                "ACTIVE": len(fields) > 1,
                "FIELDS": fields,
                "LENGTHS": {"title": 5.0, "body": 15.0, "url": 5.0, "all": 20.0},
                "WEIGHTS": {"title": 2.0, "body": 1.0, "url": 1.5, "all": 1.0},
            },
        },
        "SEMANTIC": {
            "SERVICE_URL": "http://localhost:8001",
            "SERVICE_PORT": 8001,
            "MODEL_ID": "nomic-ai/nomic-embed-text-v1.5",
            "DIM": 64,
            "DEVICE": "cpu",
            "BATCH_SIZE": 2,
        },
        "BM25": {"K1": 1.5, "B": 0.75},
        "INGESTION": {"NUM_DOCUMENTS": 5, "BATCH_SIZE": 5},
        "QUERY": {"MAX_PHRASE_LEN": 5},
        "LTR": {
            "DATA_DIR": temp_dir,
            "QUERIES": os.path.join(temp_dir, "queries.tsv"),
            "QRELS": os.path.join(temp_dir, "qrels.tsv"),
            "SPLIT_DIR": os.path.join(temp_dir, "splits"),
            "TRAIN_CACHE": os.path.join(temp_dir, "train_cache.npz"),
            "VAL_CACHE": os.path.join(temp_dir, "val_cache.npz"),
            "CANDIDATE_TOPN": 5,
            "LIST_SIZE": 3,
            "EPOCHS": 1,
            "BATCH_SIZE": 2,
            "LEARNING_RATE": 0.001,
            "SEED": 42,
            "MODEL": {
                "HIDDEN_UNITS": [16, 8],
                "DROPOUT": 0.1,
                "USE_ATTENTION": False,
            },
            "FEATURES": [
                "bm25_score",
                "query_len",
                "query_uniq_len",
                "title_len",
                "body_len",
                "title_overlap_cnt",
                "body_overlap_cnt",
                "body_overlap_ratio",
                "idf_body_overlap_sum",
                "idf_title_overlap_sum",
            ],
        },
    }
    return OmegaConf.create(config_dict)


class TestSetupMixin:
    """Mixin class providing common test setup."""

    def setup_temp_dir(self):
        """Create temporary directory with test data files."""
        self.temp_dir = tempfile.mkdtemp(prefix="sea_test_")
        self.cfg = create_test_config(self.temp_dir)

        # Write test files
        docs_path = Path(self.temp_dir) / "docs.tsv"
        docs_path.write_text(SAMPLE_DOCS_TSV)

        queries_path = Path(self.temp_dir) / "queries.tsv"
        queries_path.write_text(SAMPLE_QUERIES_TSV)

        qrels_path = Path(self.temp_dir) / "qrels.tsv"
        qrels_path.write_text(SAMPLE_QRELS_TSV)

        # Create block directories
        for field in self.cfg.SEARCH.FIELDED.FIELDS:
            os.makedirs(os.path.join(self.temp_dir, "blocks", field), exist_ok=True)

        # Create splits directory
        splits_dir = Path(self.temp_dir) / "splits"
        splits_dir.mkdir(exist_ok=True)
        (splits_dir / "train_qids.txt").write_text("Q001\nQ002\n")
        (splits_dir / "val_qids.txt").write_text("Q003\n")
        (splits_dir / "test_qids.txt").write_text("")

    def teardown_temp_dir(self):
        """Clean up temporary directory."""
        if hasattr(self, "temp_dir") and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


# =====================================================================
# STORAGE I/O TESTS
# =====================================================================


class TestStorageIO(unittest.TestCase, TestSetupMixin):
    """Test binary I/O operations for index files."""

    def setUp(self):
        self.setup_temp_dir()

    def tearDown(self):
        self.teardown_temp_dir()

    def test_block_io_write_and_read(self):
        """Test writing and reading block files."""
        from sea.storage.IO import BlockIO

        block_io = BlockIO(cfg=self.cfg, field="body")

        # Create test posting list
        test_index = {
            "apple": array("I", [0, 2, 1, 1]),  # doc0: tf=2, doc1: tf=1
            "banana": array("I", [1, 3]),  # doc1: tf=3
        }

        # Write block
        block_io.write_block("test_block", test_index)

        # Verify file exists
        block_path = os.path.join(self.temp_dir, "blocks", "body", "tokenizer_output_test_block.bin")
        self.assertTrue(os.path.exists(block_path))

        # Read and verify
        with open(block_path, "rb") as f:
            block_io.check_magic_header(f)

            term1, arr1 = block_io.read_line(f)
            self.assertEqual(term1, "apple")  # Sorted order
            self.assertEqual(list(arr1), [0, 2, 1, 1])

            term2, arr2 = block_io.read_line(f)
            self.assertEqual(term2, "banana")
            self.assertEqual(list(arr2), [1, 3])

    def test_term_dictionary_io(self):
        """Test term dictionary write and read."""
        from sea.storage.IO import TermDictionaryIO

        # Write
        term_io = TermDictionaryIO(rewrite=True, cfg=self.cfg, field="body")
        term_io.write("apple", 100, 50)
        term_io.write("banana", 150, 30)
        term_io.close()

        # Read
        term_io = TermDictionaryIO(rewrite=False, cfg=self.cfg, field="body")
        term_dict = term_io.read()
        term_io.close()

        self.assertEqual(term_dict["apple"], (100, 50))
        self.assertEqual(term_dict["banana"], (150, 30))

    def test_posting_list_io(self):
        """Test posting list write and read."""
        from sea.storage.IO import PostingListIO

        # Write
        posting_io = PostingListIO(rewrite=True, cfg=self.cfg, field="body")
        test_posting = array("I", [0, 5, 1, 3, 2, 7])
        offset, length = posting_io.write(test_posting)
        posting_io.close()

        # Read
        posting_io = PostingListIO(rewrite=False, cfg=self.cfg, field="body")
        result = posting_io.read(offset, length)
        posting_io.close()

        # First element is length, followed by data
        self.assertEqual(result[0], 6)  # Length of original array
        self.assertEqual(list(result[1:]), [0, 5, 1, 3, 2, 7])

    def test_doc_dictionary_io(self):
        """Test document dictionary write and read."""
        from sea.storage.IO import DocDictionaryIO

        # Write
        doc_io = DocDictionaryIO(rewrite=True, cfg=self.cfg, field="body")
        metadata = {
            0: ("D001", 15),
            1: ("D002", 20),
            2: ("D003", 25),
        }
        doc_io.write_metadata(metadata)
        doc_io.close()

        # Read
        doc_io = DocDictionaryIO(rewrite=False, cfg=self.cfg, field="body")
        result = doc_io.read()
        doc_io.close()

        self.assertEqual(result[0], ("D001", 15))
        self.assertEqual(result[1], ("D002", 20))
        self.assertEqual(result[2], ("D003", 25))

    def test_storage_manager_roundtrip(self):
        """Test StorageManager for writing and reading index data."""
        from sea.storage.manager import StorageManager

        # Write using storage manager
        sm_write = StorageManager(rewrite=True, cfg=self.cfg, field="body")

        test_posting1 = array("I", [0, 2, 1, 1])
        test_posting2 = array("I", [2, 3])

        sm_write.write_term_posting_list("apple", test_posting1)
        sm_write.write_term_posting_list("banana", test_posting2)
        sm_write.close()

        # Write doc metadata separately
        from sea.storage.IO import DocDictionaryIO

        doc_io = DocDictionaryIO(rewrite=True, cfg=self.cfg, field="body")
        doc_io.write_metadata({0: ("D001", 10), 1: ("D002", 15), 2: ("D003", 20)})
        doc_io.close()

        # Read back
        sm_read = StorageManager(rewrite=False, cfg=self.cfg, field="body")

        term_dict = sm_read.getTermDictionary()
        self.assertIn("apple", term_dict)
        self.assertIn("banana", term_dict)

        posting = sm_read.getPostingList("apple")
        self.assertIsNotNone(posting)
        # Posting list format: [count, data...]
        self.assertEqual(posting[0], 4)  # Length

        doc_meta = sm_read.getDocMetadata()
        self.assertEqual(doc_meta[0][0], "D001")

        sm_read.close()


# =====================================================================
# EMBEDDING STORAGE TESTS
# =====================================================================


class TestEmbeddingStorage(unittest.TestCase, TestSetupMixin):
    """Test embedding binary storage."""

    def setUp(self):
        self.setup_temp_dir()

    def tearDown(self):
        self.teardown_temp_dir()

    def test_embedding_io_write_and_read(self):
        """Test writing and reading embedding files."""
        from sea.storage.embeddings import EmbeddingIO

        emb_io = EmbeddingIO(cfg=self.cfg)

        # Create test embeddings (5 docs, 64 dims)
        embeddings = np.random.randn(5, 64).astype(np.float32)
        # Normalize like the real system does
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        emb_io.write(embeddings, force=True)

        # Read back
        emb_io2 = EmbeddingIO(cfg=self.cfg)
        loaded = emb_io2.load_all()

        self.assertEqual(loaded.shape, (5, 64))
        np.testing.assert_allclose(loaded, embeddings, rtol=1e-5)

        # Test single embedding access
        single = emb_io2.get_embedding(2)
        np.testing.assert_allclose(single, embeddings[2], rtol=1e-5)

        emb_io2.close()

    def test_embedding_io_properties(self):
        """Test embedding IO properties."""
        from sea.storage.embeddings import EmbeddingIO

        emb_io = EmbeddingIO(cfg=self.cfg)
        embeddings = np.random.randn(5, 64).astype(np.float32)
        emb_io.write(embeddings, force=True)

        emb_io2 = EmbeddingIO(cfg=self.cfg)
        self.assertEqual(emb_io2.num_docs, 5)
        self.assertEqual(emb_io2.dim, 64)
        emb_io2.close()


# =====================================================================
# INGESTION PIPELINE TESTS
# =====================================================================


class TestIngestionPipeline(unittest.TestCase, TestSetupMixin):
    """Test document ingestion from TSV to binary index."""

    def setUp(self):
        self.setup_temp_dir()

    def tearDown(self):
        self.teardown_temp_dir()

    def test_worker_process_batch(self):
        """Test single worker batch processing."""
        from sea.ingest.worker import Worker

        # Parse documents
        lines = []
        for i, line in enumerate(SAMPLE_DOCS_TSV.strip().split("\n")):
            lines.append((i, line + "\n"))

        worker = Worker(
            store_positions=False,
            cfg=self.cfg,
            fields=["body"],
        )

        result = worker.process_batch("0", lines)

        # Check metadata was captured
        self.assertIn("body", result.metadata)
        self.assertEqual(len(result.metadata["body"]), 5)

        # Check block file was created (use cfg.BLOCK_PATH for correct path)
        block_path = os.path.join(self.cfg.BLOCK_PATH, "body", "tokenizer_output_0.bin")
        self.assertTrue(os.path.exists(block_path))

        # Verify block contains expected terms
        from sea.storage.IO import BlockIO

        block_io = BlockIO(cfg=self.cfg, field="body")
        found_terms = set()
        with open(block_path, "rb") as f:
            block_io.check_magic_header(f)
            while True:
                term, arr = block_io.read_line(f)
                if term is None:
                    break
                found_terms.add(term)

        # Check that key terms were indexed
        self.assertIn("apple", found_terms)
        self.assertIn("banana", found_terms)
        self.assertIn("fruit", found_terms)

    def test_kmerger_merge_blocks(self):
        """Test k-way merging of block files."""
        from sea.ingest.worker import Worker
        from sea.ingest.kmerger import KMerger
        from sea.storage.IO import DocDictionaryIO

        # First create blocks
        lines = [(i, line + "\n") for i, line in enumerate(SAMPLE_DOCS_TSV.strip().split("\n"))]

        # Split into 2 batches to test merging
        batch1 = lines[:3]
        batch2 = lines[3:]

        worker = Worker(store_positions=False, cfg=self.cfg, fields=["body"])

        result1 = worker.process_batch("0", batch1)
        result2 = worker.process_batch("1", batch2)

        # Write document metadata
        combined_metadata = {**result1.metadata["body"], **result2.metadata["body"]}
        doc_io = DocDictionaryIO(rewrite=True, cfg=self.cfg, field="body")
        doc_io.write_metadata(combined_metadata)
        doc_io.close()

        # Merge blocks
        merger = KMerger(self.cfg.BLOCK_PATH, fields=["body"], cfg=self.cfg)
        num_terms, _ = merger.merge_blocks()

        self.assertGreater(num_terms, 0)

        # Verify merged index can be read
        from sea.storage.manager import StorageManager

        sm = StorageManager(rewrite=False, cfg=self.cfg, field="body")
        term_dict = sm.getTermDictionary()

        self.assertIn("apple", term_dict)
        self.assertIn("banana", term_dict)

        # Check posting list for 'apple'
        posting = sm.getPostingList("apple")
        self.assertIsNotNone(posting)
        self.assertGreater(len(posting), 0)

        sm.close()


# =====================================================================
# BM25 RANKING TESTS
# =====================================================================


class TestBM25Ranking(unittest.TestCase, TestSetupMixin):
    """Test BM25 ranking on indexed documents."""

    def setUp(self):
        self.setup_temp_dir()
        self._build_test_index()

    def tearDown(self):
        self.teardown_temp_dir()

    def _build_test_index(self):
        """Build a minimal index for testing."""
        from sea.ingest.worker import Worker
        from sea.ingest.kmerger import KMerger
        from sea.storage.IO import DocDictionaryIO

        # Use single field for simpler testing (use "body" since "all" has enum bug)
        # FIELDED.ACTIVE must be True so RankerAdapter uses FIELDED.FIELDS
        self.cfg.SEARCH.FIELDED.ACTIVE = True
        self.cfg.SEARCH.FIELDED.FIELDS = ["body"]

        lines = [(i, line + "\n") for i, line in enumerate(SAMPLE_DOCS_TSV.strip().split("\n"))]

        os.makedirs(os.path.join(self.temp_dir, "blocks", "body"), exist_ok=True)

        worker = Worker(store_positions=False, cfg=self.cfg, fields=["body"])
        result = worker.process_batch("0", lines)

        # Write doc metadata
        doc_io = DocDictionaryIO(rewrite=True, cfg=self.cfg, field="body")
        doc_io.write_metadata(result.metadata["body"])
        doc_io.close()

        # Merge
        merger = KMerger(self.cfg.BLOCK_PATH, fields=["body"], cfg=self.cfg)
        merger.merge_blocks()

    def test_bm25_ranking_algorithm(self):
        """Test BM25 scoring produces expected ranking order."""
        from sea.ranking.ranking import BM25Ranking

        ranking = BM25Ranking(self.cfg)

        # Simulate token data: {doc_id: (doc_len, tf)}
        token_data = {
            0: (20, 3),  # doc0: len=20, tf=3
            1: (15, 1),  # doc1: len=15, tf=1
            2: (25, 2),  # doc2: len=25, tf=2
        }

        scores = ranking._compute_score(token_data, field="body")

        # Higher TF should generally produce higher score (adjusted for doc length)
        self.assertIn(0, scores)
        self.assertIn(1, scores)
        self.assertIn(2, scores)
        # Doc 0 has highest TF, should have highest or near-highest score
        self.assertGreater(scores[0], scores[1])

    def test_bm25_ranker_adapter(self):
        """Test full BM25 ranking pipeline with RankerAdapter."""
        from sea.ranking.io_wrapper import bm25
        from sea.index.tokenization import get_tokenizer

        ranker = bm25(cfg=self.cfg)
        tokenizer = get_tokenizer(self.cfg)

        # Search for "apple"
        tokens = tokenizer.tokenize("apple fruit")
        results = ranker(tokens)

        # Should return documents
        self.assertGreater(len(results), 0)

        # Results should have Document objects with scores
        self.assertTrue(hasattr(results[0], "score"))
        self.assertTrue(hasattr(results[0], "doc_id"))
        self.assertTrue(hasattr(results[0], "content"))

        # Scores should be positive
        self.assertGreater(results[0].score, 0)

    def test_bm25_retriever_wrapper(self):
        """Test BM25Retriever wrapper used in LTR pipeline."""
        from sea.ltr.bm25 import BM25Retriever

        retriever = BM25Retriever.from_config(self.cfg)

        # Test retrieve with documents
        docs = retriever.retrieve("apple fruit", topn=3)
        self.assertGreater(len(docs), 0)
        self.assertLessEqual(len(docs), 3)

        # Test retrieve_ids (without hydration)
        id_results = retriever.retrieve_ids("banana", topn=3)
        self.assertGreater(len(id_results), 0)
        self.assertIsInstance(id_results[0], tuple)
        self.assertEqual(len(id_results[0]), 2)  # (doc_id, score)


# =====================================================================
# SEMANTIC SEARCH TESTS
# =====================================================================


class TestSemanticSearch(unittest.TestCase, TestSetupMixin):
    """Test semantic search with embeddings."""

    def setUp(self):
        self.setup_temp_dir()
        self._create_test_embeddings()

    def tearDown(self):
        self.teardown_temp_dir()

    def _create_test_embeddings(self):
        """Create test embeddings for 5 documents."""
        from sea.storage.embeddings import EmbeddingIO

        # Create normalized embeddings
        np.random.seed(42)
        embeddings = np.random.randn(5, 64).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        emb_io = EmbeddingIO(cfg=self.cfg)
        emb_io.write(embeddings, force=True)

        self.test_embeddings = embeddings

    @patch("sea.semantic.search.EmbeddingClient")
    def test_semantic_searcher(self, mock_client_cls):
        """Test SemanticSearcher with mocked embedding client."""
        from sea.semantic.search import SemanticSearcher

        # Mock the client to return a query embedding
        mock_client = MagicMock()
        # Return an embedding similar to doc 0
        query_emb = self.test_embeddings[0] + np.random.randn(64).astype(np.float32) * 0.1
        query_emb = query_emb / np.linalg.norm(query_emb)
        mock_client.embed_query.return_value = query_emb
        mock_client_cls.return_value = mock_client

        searcher = SemanticSearcher(cfg=self.cfg, verbose=False)

        results = searcher.search("test query", topn=3)

        self.assertEqual(len(results), 3)
        # Results should be (doc_id, score) tuples
        self.assertIsInstance(results[0][0], int)
        self.assertIsInstance(results[0][1], float)

        # Doc 0 should be among top results since query is similar
        top_doc_ids = [r[0] for r in results]
        self.assertIn(0, top_doc_ids)


# =====================================================================
# LTR DATA PREPARATION TESTS
# =====================================================================


class TestLTRDataPreparation(unittest.TestCase, TestSetupMixin):
    """Test LTR feature extraction and data preparation."""

    def setUp(self):
        self.setup_temp_dir()
        self._build_test_index()

    def tearDown(self):
        self.teardown_temp_dir()

    def _build_test_index(self):
        """Build minimal index for LTR testing."""
        from sea.ingest.worker import Worker
        from sea.ingest.kmerger import KMerger
        from sea.storage.IO import DocDictionaryIO

        # Use "body" field since "all" has enum bug
        # FIELDED.ACTIVE must be True so RankerAdapter uses FIELDED.FIELDS
        self.cfg.SEARCH.FIELDED.ACTIVE = True
        self.cfg.SEARCH.FIELDED.FIELDS = ["body"]

        lines = [(i, line + "\n") for i, line in enumerate(SAMPLE_DOCS_TSV.strip().split("\n"))]
        os.makedirs(os.path.join(self.temp_dir, "blocks", "body"), exist_ok=True)

        worker = Worker(store_positions=False, cfg=self.cfg, fields=["body"])
        result = worker.process_batch("0", lines)

        doc_io = DocDictionaryIO(rewrite=True, cfg=self.cfg, field="body")
        doc_io.write_metadata(result.metadata["body"])
        doc_io.close()

        merger = KMerger(self.cfg.BLOCK_PATH, fields=["body"], cfg=self.cfg)
        merger.merge_blocks()

    def test_feature_extractor(self):
        """Test feature extraction for query-document pairs."""
        from sea.ltr.features import FeatureExtractor
        from sea.ranking.utils import Document

        fe = FeatureExtractor.from_config(self.cfg, cache_max_docs=5)

        doc = Document(
            doc_id="D001",
            link="http://example.com",
            title="Apple Fruit Guide",
            content="Apples are delicious fruits",
            score=1.5,
        )

        features = fe.extract("apple fruit", doc)

        self.assertEqual(features.dtype, np.float32)
        self.assertEqual(len(features), len(fe.features.names))
        self.assertTrue(np.isfinite(features).all())

    def test_feature_extractor_batch(self):
        """Test batch feature extraction."""
        from sea.ltr.features import FeatureExtractor
        from sea.ranking.utils import Document

        fe = FeatureExtractor.from_config(self.cfg, cache_max_docs=5)

        docs = [
            Document("D001", "http://a.com", "Apple Guide", "Apples are fruits", 1.5),
            Document("D002", "http://b.com", "Banana Info", "Bananas are yellow", 1.2),
        ]

        features = fe.extract_many("fruit apple banana", docs)

        self.assertEqual(features.shape, (2, len(fe.features.names)))
        self.assertTrue(np.isfinite(features).all())

    def test_sample_list_for_query(self):
        """Test sampling positive and negative documents for a query."""
        from sea.ltr.prepare_data import _sample_list_for_query
        from sea.ltr.bm25 import BM25Retriever
        from sea.ltr.features import FeatureExtractor

        retriever = BM25Retriever.from_config(self.cfg)
        fe = FeatureExtractor.from_config(self.cfg)

        # Get candidates
        id_results = retriever.retrieve_ids("apple fruit", topn=5)

        if len(id_results) < 2:
            self.skipTest("Not enough documents indexed for this test")

        positives_internal = {id_results[0][0]}  # First result is "positive"

        result = _sample_list_for_query(
            qid=1,
            query="apple fruit",
            id_results=id_results,
            positives_internal=positives_internal,
            retriever=retriever,
            fe=fe,
            list_size=3,
            seed=42,
        )

        if result is None:
            self.skipTest("No valid sample could be created")

        features, labels = result
        self.assertEqual(features.shape[0], 3)  # list_size
        self.assertEqual(labels.shape[0], 3)
        self.assertEqual(labels.sum(), 1.0)  # Exactly one positive


# =====================================================================
# LTR MODEL TRAINING TESTS
# =====================================================================


class TestLTRModelTraining(unittest.TestCase, TestSetupMixin):
    """Test LTR model training with minimal data."""

    def setUp(self):
        self.setup_temp_dir()

    def tearDown(self):
        self.teardown_temp_dir()

    def test_tfr_model_build_and_compile(self):
        """Test TFR model construction and compilation."""
        from sea.ltr.tfr_model import TFRConfig, build_tfr_scoring_model, compile_tfr_model

        config = TFRConfig(
            list_size=5,
            num_features=10,
            learning_rate=0.001,
            hidden_units=(16, 8),
            dropout=0.1,
            use_attention=False,
        )

        model = build_tfr_scoring_model(config)
        self.assertIsNotNone(model)

        # Check input/output shapes
        self.assertEqual(model.input_shape, (None, 5, 10))
        self.assertEqual(model.output_shape, (None, 5))

        model = compile_tfr_model(model, learning_rate=0.001)
        self.assertIsNotNone(model.optimizer)

    def test_tfr_model_training_step(self):
        """Test one training step with synthetic data."""
        from sea.ltr.tfr_model import TFRConfig, build_tfr_scoring_model, compile_tfr_model
        import tensorflow as tf

        config = TFRConfig(
            list_size=5,
            num_features=10,
            learning_rate=0.001,
            hidden_units=(16, 8),
            dropout=0.1,
            use_attention=False,
        )

        model = build_tfr_scoring_model(config)

        # Adapt normalization layer with sample data
        sample_features = np.random.randn(10, 5, 10).astype(np.float32)
        model.get_layer("normalization").adapt(sample_features)

        model = compile_tfr_model(model, learning_rate=0.001)

        # Create synthetic batch
        X = np.random.randn(4, 5, 10).astype(np.float32)  # 4 samples
        y = np.zeros((4, 5), dtype=np.float32)
        y[:, 0] = 1.0  # First doc is relevant

        # One training step
        history = model.fit(X, y, epochs=1, verbose=0)

        self.assertIn("loss", history.history)
        self.assertEqual(len(history.history["loss"]), 1)

    def test_cache_loading_and_training(self):
        """Test training from pre-computed cache file."""
        from sea.ltr.tfr_model import TFRConfig, build_tfr_scoring_model, compile_tfr_model

        # Create synthetic cache
        num_samples = 10
        list_size = 5
        num_features = 10

        features = np.random.randn(num_samples, list_size, num_features).astype(np.float32)
        labels = np.zeros((num_samples, list_size), dtype=np.float32)
        labels[:, 0] = 1.0  # First position is always relevant

        cache_path = os.path.join(self.temp_dir, "test_cache.npz")
        np.savez_compressed(cache_path, features=features, labels=labels)

        # Load and verify
        loaded = np.load(cache_path)
        self.assertEqual(loaded["features"].shape, (num_samples, list_size, num_features))
        self.assertEqual(loaded["labels"].shape, (num_samples, list_size))

        # Build and train model
        config = TFRConfig(
            list_size=list_size,
            num_features=num_features,
            learning_rate=0.001,
            hidden_units=(16, 8),
            dropout=0.1,
            use_attention=False,
        )

        model = build_tfr_scoring_model(config)
        model.get_layer("normalization").adapt(loaded["features"][:5])
        model = compile_tfr_model(model, learning_rate=0.001)

        history = model.fit(
            loaded["features"],
            loaded["labels"],
            epochs=2,
            batch_size=4,
            verbose=0,
        )

        self.assertEqual(len(history.history["loss"]), 2)
        # Verify training ran (losses are computed)
        self.assertTrue(all(np.isfinite(history.history["loss"])))


# =====================================================================
# INTEGRATION TESTS - FULL PIPELINE
# =====================================================================


class TestFullPipeline(unittest.TestCase, TestSetupMixin):
    """Full integration tests from ingestion to search."""

    def setUp(self):
        self.setup_temp_dir()

    def tearDown(self):
        self.teardown_temp_dir()

    def test_ingestion_to_search_pipeline(self):
        """Test complete pipeline: ingest -> index -> search."""
        from sea.ingest.worker import Worker
        from sea.ingest.kmerger import KMerger
        from sea.storage.IO import DocDictionaryIO
        from sea.ltr.bm25 import BM25Retriever

        # Configure for single field (use "body" since "all" has enum bug)
        # FIELDED.ACTIVE must be True so RankerAdapter uses FIELDED.FIELDS
        self.cfg.SEARCH.FIELDED.ACTIVE = True
        self.cfg.SEARCH.FIELDED.FIELDS = ["body"]

        # Step 1: Ingest documents
        lines = [(i, line + "\n") for i, line in enumerate(SAMPLE_DOCS_TSV.strip().split("\n"))]
        os.makedirs(os.path.join(self.temp_dir, "blocks", "body"), exist_ok=True)

        worker = Worker(store_positions=False, cfg=self.cfg, fields=["body"])
        result = worker.process_batch("0", lines)

        # Step 2: Write metadata
        doc_io = DocDictionaryIO(rewrite=True, cfg=self.cfg, field="body")
        doc_io.write_metadata(result.metadata["body"])
        doc_io.close()

        # Step 3: Merge blocks
        merger = KMerger(self.cfg.BLOCK_PATH, fields=["body"], cfg=self.cfg)
        num_terms, _ = merger.merge_blocks()
        self.assertGreater(num_terms, 0)

        # Step 4: Search
        retriever = BM25Retriever.from_config(self.cfg)

        # Search for "apple pie"
        docs = retriever.retrieve("apple pie", topn=3)
        self.assertGreater(len(docs), 0)

        # The recipe document should be in top results
        doc_ids = [d.doc_id for d in docs]
        # D001 (Apple Fruit Guide) or D003 (Recipe Collection) should be high
        self.assertTrue(any(did in ["D001", "D003"] for did in doc_ids))

        # Search for "banana"
        docs = retriever.retrieve("banana", topn=3)
        self.assertGreater(len(docs), 0)
        doc_ids = [d.doc_id for d in docs]
        # Documents with "banana" in body should be returned
        # D002, D003, D004, D005 all contain "banana"
        banana_docs = {"D002", "D003", "D004", "D005"}
        self.assertTrue(
            any(did in banana_docs for did in doc_ids),
            f"Expected at least one banana doc in results, got: {doc_ids}"
        )

    def test_fielded_search_pipeline(self):
        """Test fielded search with multiple fields."""
        from sea.ingest.worker import Worker
        from sea.ingest.kmerger import KMerger
        from sea.storage.IO import DocDictionaryIO
        from sea.ltr.bm25 import BM25Retriever

        # Configure for multiple fields
        self.cfg.SEARCH.FIELDED.ACTIVE = True
        self.cfg.SEARCH.FIELDED.FIELDS = ["title", "body"]

        lines = [(i, line + "\n") for i, line in enumerate(SAMPLE_DOCS_TSV.strip().split("\n"))]

        for field in ["title", "body"]:
            os.makedirs(os.path.join(self.temp_dir, "blocks", field), exist_ok=True)

        worker = Worker(store_positions=False, cfg=self.cfg, fields=["title", "body"])
        result = worker.process_batch("0", lines)

        # Write metadata for each field
        for field in ["title", "body"]:
            doc_io = DocDictionaryIO(rewrite=True, cfg=self.cfg, field=field)
            doc_io.write_metadata(result.metadata[field])
            doc_io.close()

        # Merge blocks
        merger = KMerger(self.cfg.BLOCK_PATH, fields=["title", "body"], cfg=self.cfg)
        merger.merge_blocks()

        # Search with fielded retrieval
        retriever = BM25Retriever.from_config(self.cfg)
        docs = retriever.retrieve("apple guide", topn=3)

        self.assertGreater(len(docs), 0)
        # Title field has higher weight, so "Apple Fruit Guide" (D001) should rank high
        self.assertEqual(docs[0].doc_id, "D001")


# =====================================================================
# EMBEDDING SERVICE TESTS (with mocking)
# =====================================================================


class TestEmbeddingService(unittest.TestCase, TestSetupMixin):
    """Test embedding service client and integration."""

    def setUp(self):
        self.setup_temp_dir()

    def tearDown(self):
        self.teardown_temp_dir()

    @patch("sea.semantic.client.httpx.Client")
    def test_embedding_client(self, mock_httpx_client):
        """Test EmbeddingClient with mocked HTTP."""
        from sea.semantic.client import EmbeddingClient

        # Create mock embeddings (1 query -> 1 embedding)
        mock_embeddings = np.random.randn(1, 64).astype(np.float32).tolist()

        # Mock response
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"embeddings": mock_embeddings}

        # Setup context manager mock
        mock_client_instance = MagicMock()
        mock_client_instance.post.return_value = mock_response
        mock_httpx_client.return_value.__enter__.return_value = mock_client_instance
        mock_httpx_client.return_value.__exit__.return_value = False

        client = EmbeddingClient(base_url="http://localhost:8001")
        result = client.embed_query("test query")

        self.assertEqual(len(result), 64)
        self.assertEqual(result.dtype, np.float32)
        mock_client_instance.post.assert_called_once()


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from sea.ltr.serve_tfr import TFRReranker
from sea.ranking.utils import Document

class TestLTRServing(unittest.TestCase):
    @patch('tensorflow.keras.models.load_model')
    @patch('sea.ltr.bm25.BM25Retriever.from_config')
    @patch('sea.ltr.features.FeatureExtractor.from_config')
    def test_reranker_rerank(self, mock_fe_from_cfg, mock_retriever_from_cfg, mock_load_model):
        # Setup mocks
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        
        mock_retriever = MagicMock()
        mock_retriever_from_cfg.return_value = mock_retriever
        
        mock_fe = MagicMock()
        mock_fe_from_cfg.return_value = mock_fe
        
        # Mock retriever.retrieve to return some docs
        doc1 = Document(doc_id="D1", link="", title="Title 1", content="Content 1", score=1.0)
        doc2 = Document(doc_id="D2", link="", title="Title 2", content="Content 2", score=0.5)
        mock_retriever.retrieve.return_value = [doc1, doc2]
        
        # Mock fe.extract_many to return some features
        mock_fe.extract_many.return_value = np.zeros((2, 5))
        
        # Mock model.predict to return scores
        # scores for doc1 and doc2
        mock_model.predict.return_value = np.array([[0.1, 0.9]])
        
        reranker = TFRReranker.load(model_path="fake/path")
        
        results = reranker.rerank("test query", candidate_topn=2, topk=2)
        
        self.assertEqual(len(results), 2)
        # doc2 should be first because it got higher score (0.9) from model.predict
        self.assertEqual(results[0].doc_id, "D2")
        self.assertEqual(results[0].score, 0.9)
        self.assertEqual(results[1].doc_id, "D1")
        self.assertEqual(results[1].score, 0.1)

    @patch('tensorflow.keras.models.load_model')
    @patch('sea.ltr.bm25.BM25Retriever.from_config')
    @patch('sea.ltr.features.FeatureExtractor.from_config')
    def test_reranker_no_docs(self, mock_fe_from_cfg, mock_retriever_from_cfg, mock_load_model):
        mock_retriever = MagicMock()
        mock_retriever_from_cfg.return_value = mock_retriever
        mock_retriever.retrieve.return_value = []
        
        reranker = TFRReranker.load(model_path="fake/path")
        results = reranker.rerank("test query")
        
        self.assertEqual(results, [])

if __name__ == "__main__":
    unittest.main()


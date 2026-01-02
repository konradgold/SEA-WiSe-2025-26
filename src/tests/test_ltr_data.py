import unittest
from unittest.mock import MagicMock
import numpy as np
from sea.ltr.tfr_data import _sample_list_for_query, ListwiseSample

class TestLTRData(unittest.TestCase):
    def test_sample_list_for_query(self):
        # Mocking retriever and its dependencies
        mock_retriever = MagicMock()
        mock_retriever.ranker.storage_manager.getDocMetadataEntry.side_effect = lambda int_id: (f"D{int_id}", None)
        
        # Mock hydrate_docs to return Document objects
        def mock_hydrate(id_score_pairs):
            docs = []
            for int_id, score in id_score_pairs:
                doc = MagicMock()
                doc.doc_id = f"D{int_id}"
                doc.score = score
                docs.append(doc)
            return docs
        mock_retriever.hydrate_docs.side_effect = mock_hydrate

        # Mock feature extractor
        mock_fe = MagicMock()
        mock_fe.features.names = ["f1", "f2"]
        mock_fe.extract_many.side_effect = lambda query, docs: np.zeros((len(docs), 2))

        qid = 1
        query = "test query"
        id_results = [(10, 1.0), (20, 0.5), (30, 0.3)]
        positives = {"D10"}
        list_size = 3
        
        sample = _sample_list_for_query(
            qid=qid,
            query=query,
            id_results=id_results,
            positives=positives,
            retriever=mock_retriever,
            fe=mock_fe,
            list_size=list_size,
            seed=42
        )
        
        self.assertIsInstance(sample, ListwiseSample)
        self.assertEqual(sample.qid, qid)
        self.assertEqual(sample.features.shape, (3, 2))
        self.assertEqual(sample.labels.shape, (3,))
        self.assertEqual(sample.labels.sum(), 1.0)  # Exactly one positive

    def test_sample_list_for_query_no_positives(self):
        mock_retriever = MagicMock()
        mock_retriever.ranker.storage_manager.getDocMetadataEntry.side_effect = lambda int_id: (f"D{int_id}", None)
        
        mock_fe = MagicMock()
        
        qid = 1
        query = "test query"
        id_results = [(10, 1.0), (20, 0.5)]
        positives = {"D99"}  # Not in id_results
        
        sample = _sample_list_for_query(
            qid=qid,
            query=query,
            id_results=id_results,
            positives=positives,
            retriever=mock_retriever,
            fe=mock_fe,
            list_size=3,
            seed=42
        )
        
        self.assertIsNone(sample)

if __name__ == "__main__":
    unittest.main()


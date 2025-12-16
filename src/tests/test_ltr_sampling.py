import unittest

from sea.ltr.candidates import CandidateDoc
from sea.ltr.sampling import sample_hard_negatives


class TestLTRSampling(unittest.TestCase):
    def test_sampling_excludes_positives_from_negatives(self):
        qid = 1
        candidates = [
            CandidateDoc(docid="D_pos", bm25=10.0),
            CandidateDoc(docid="D_neg1", bm25=9.0),
            CandidateDoc(docid="D_neg2", bm25=8.0),
        ]
        positives = {"D_pos"}
        samples = sample_hard_negatives(
            qid=qid,
            candidates=candidates,
            positives=positives,
            neg_per_pos=5,
            seed=42,
            hard_pool_topk=2,
        )
        self.assertTrue(len(samples) > 0)
        for s in samples:
            self.assertEqual(s.pos_docid, "D_pos")
            self.assertNotEqual(s.neg_docid, "D_pos")


if __name__ == "__main__":
    unittest.main()





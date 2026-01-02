import unittest

import numpy as np

from sea.ltr.features import FeatureExtractor
from sea.ranking.utils import Document
from sea.utils.config import Config


class TestLTRFeatures(unittest.TestCase):
    def test_feature_vector_shape_and_finite(self):
        cfg = Config(load=True)
        fe = FeatureExtractor.from_config(cfg, cache_max_docs=2)

        doc = Document(doc_id="D0", link="", title="Apple banana", content="Apple pie recipe", score=1.23)
        x = fe.extract("apple pie", doc)

        self.assertEqual(x.dtype, np.float32)
        self.assertEqual(x.shape, (len(fe.features.names),))
        self.assertTrue(np.isfinite(x).all())


if __name__ == "__main__":
    unittest.main()





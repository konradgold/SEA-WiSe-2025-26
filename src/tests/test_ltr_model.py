import unittest
import tensorflow as tf
import numpy as np
from sea.ltr.tfr_model import TFRConfig, build_tfr_scoring_model, compile_tfr_model

class TestLTRModel(unittest.TestCase):
    def test_build_model(self):
        cfg = TFRConfig(list_size=10, num_features=5)
        model = build_tfr_scoring_model(cfg)
        
        # Check input shape
        self.assertEqual(model.input_shape, (None, 10, 5))
        # Check output shape (listwise scores)
        self.assertEqual(model.output_shape, (None, 10))
        
        self.assertEqual(model.name, "tfr_reranker")

    def test_build_model_no_attention(self):
        cfg = TFRConfig(list_size=10, num_features=5, use_attention=False)
        model = build_tfr_scoring_model(cfg)
        
        # Check that no self_attention layer exists
        layer_names = [l.name for l in model.layers]
        self.assertNotIn("self_attention", layer_names)

    def test_compile_model(self):
        cfg = TFRConfig(list_size=10, num_features=5)
        model = build_tfr_scoring_model(cfg)
        model = compile_tfr_model(model)
        
        self.assertIsNotNone(model.optimizer)
        self.assertIsNotNone(model.loss)
        
        x = np.zeros((1, 10, 5), dtype=np.float32)
        y = np.zeros((1, 10), dtype=np.float32)
        model.train_on_batch(x, y)
        
        # Check if metrics are present
        metric_names = [m.name for m in model.metrics]
        self.assertIn("mrr@10", metric_names)
        self.assertIn("ndcg@10", metric_names)

if __name__ == "__main__":
    unittest.main()


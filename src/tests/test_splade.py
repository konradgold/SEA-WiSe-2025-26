import unittest
from sea.utils.config_wrapper import Config
from sea.query.splade import SpladeEncoder

class TestSplade(unittest.TestCase):
    
    def setUp(self):
        cfg = Config(load=True)
        self.test_texts = [
            "This is a test sentence for SPLADE encoding.",
            "Machine learning models process natural language."
        ]
        self.encoder = SpladeEncoder(cfg=cfg)
    
    def test_encode_returns_dicts(self):
        """Test that encode returns two dictionaries with expected structure."""
        sparse_dict, sparse_dict_tokens = self.encoder._encode(self.test_texts[0])
        
        self.assertIsInstance(sparse_dict, dict)
        self.assertIsInstance(sparse_dict_tokens, dict)
        self.assertGreater(len(sparse_dict), 0)
        self.assertGreater(len(sparse_dict_tokens), 0)
        self.assertEqual(len(sparse_dict), len(sparse_dict_tokens))

        # Check token dictionary has string keys and float values
        self.assertTrue(all(isinstance(k, str) for k in sparse_dict_tokens.keys()))
        self.assertTrue(all(isinstance(v, float) for v in sparse_dict_tokens.values()))

        # Check that all weights are greater than 0
        self.assertTrue(all(v > 0 for v in sparse_dict_tokens.values()))
        self.assertTrue(all(v > 0 for v in sparse_dict.values()))

    def test_expand_returns_list_of_tokens(self):
        """Test that expand returns a list of tokens."""
        tokens = self.encoder.expand(self.test_texts[1])
        
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)
        self.assertTrue(all(isinstance(token, str) for token in tokens))
    
    def test_tokenize(self):
        tokens = self.encoder.tokenize(self.test_texts[0])
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)
        self.assertTrue(all(isinstance(token, str) for token in tokens))




if __name__ == "__main__":
    unittest.main()



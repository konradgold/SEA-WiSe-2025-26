from email.mime import base
import unittest

import yaml
from sea.utils.config import Config


class TestConfig(unittest.TestCase):
    def test_parse_base_config(self):
        cfg = Config(load=True)
        self.assertIsNotNone(cfg)
        base_path = "configs/base.yaml"
        with open(base_path, 'r') as f:
            yaml_config = yaml.load(f.read(), Loader=yaml.SafeLoader)
        
        self.assertEqual(cfg.REDIS_HOST, yaml_config['REDIS_HOST'])
        self.assertEqual(cfg.REDIS_PORT, yaml_config['REDIS_PORT'])
        self.assertEqual(cfg.TOKENIZER.BACKEND, yaml_config['TOKENIZER']['BACKEND'])

    def test_update_config(self):
        cfg = Config(load=True)
        self.assertIsNotNone(cfg)
        new_config = {
            'REDIS_HOST': 'not_localhost',
            'TOKENIZER': {
                'BACKEND': 'new-tokenizer-model'
            }
        }
        self.assertIsNotNone(cfg.TOKENIZER.MIN_LEN)
        cfg.update_dict(new_config)
        self.assertEqual(cfg.REDIS_HOST, 'not_localhost')
        self.assertEqual(cfg.TOKENIZER.BACKEND, 'new-tokenizer-model')
        self.assertEqual(cfg.REDIS_PORT, 6379)  # unchangeds
        self.assertIsNone(cfg.TOKENIZER.MIN_LEN)
        
if __name__ == "__main__":
    unittest.main()


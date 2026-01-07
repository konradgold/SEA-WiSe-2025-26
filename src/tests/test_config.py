import unittest

import yaml
from sea.utils.config_wrapper import Config


class TestConfig(unittest.TestCase):
    def test_parse_base_config(self):
        cfg = Config(load=True)
        self.assertIsNotNone(cfg)
        base_path = "configs/base.yaml"
        with open(base_path, 'r') as f:
            yaml_config = yaml.load(f.read(), Loader=yaml.SafeLoader)

        self.assertEqual(cfg.TOKENIZER.BACKEND, yaml_config['TOKENIZER']['BACKEND'])
        self.assertEqual(cfg.DOCUMENTS, yaml_config["DOCUMENTS"])
        self.assertEqual(cfg.SEARCH.POSTINGS_CUT, yaml_config["SEARCH"]["POSTINGS_CUT"])

    def test_update_config(self):
        cfg = Config(load=True)
        self.assertIsNotNone(cfg)
        new_config = {"TOKENIZER": {"BACKEND": "new-tokenizer-model"}}
        cfg.TOKENIZER.BACKEND = new_config["TOKENIZER"]["BACKEND"]
        cfg.TOKENIZER.MIN_LEN = 2
        self.assertIsNotNone(cfg.TOKENIZER.MIN_LEN)
        self.assertEqual(cfg.TOKENIZER.BACKEND, "new-tokenizer-model")
        self.assertEqual(cfg.TOKENIZER.MIN_LEN, 2)

if __name__ == "__main__":
    unittest.main()

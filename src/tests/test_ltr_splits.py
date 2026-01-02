import tempfile
import unittest
from pathlib import Path

from sea.ltr.msmarco import QuerySplit, make_query_split, persist_query_split


class TestLTRSplits(unittest.TestCase):
    def test_query_split_is_disjoint_and_complete(self):
        qids = list(range(1, 101))
        split: QuerySplit = make_query_split(qids, seed=123, train_frac=0.8, val_frac=0.1, test_frac=0.1)
        self.assertEqual(len(split.train) + len(split.val) + len(split.test), 100)
        self.assertEqual(len(set(split.train) & set(split.val)), 0)
        self.assertEqual(len(set(split.train) & set(split.test)), 0)
        self.assertEqual(len(set(split.val) & set(split.test)), 0)

    def test_split_persist_roundtrip_files_exist(self):
        qids = list(range(1, 51))
        split = make_query_split(qids, seed=1)
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            persist_query_split(split, out_dir=out)
            self.assertTrue((out / "train_qids.txt").exists())
            self.assertTrue((out / "val_qids.txt").exists())
            self.assertTrue((out / "test_qids.txt").exists())
            self.assertTrue((out / "split_meta.json").exists())


if __name__ == "__main__":
    unittest.main()





from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from sea.ltr.bm25 import BM25Retriever
from sea.ltr.features import FeatureExtractor
from sea.ranking.utils import Document
from sea.utils.config import Config


@dataclass
class TFRReranker:
    cfg: Config
    retriever: BM25Retriever
    fe: FeatureExtractor
    model: object

    @classmethod
    def load(cls, *, model_path: str | Path, cfg: Optional[Config] = None) -> "TFRReranker":
        cfg = cfg or Config(load=True)
        retriever = BM25Retriever.from_config(cfg)
        fe = FeatureExtractor.from_config(cfg)

        import tensorflow as tf

        model = tf.keras.models.load_model(str(model_path), compile=False)
        return cls(cfg=cfg, retriever=retriever, fe=fe, model=model)

    def rerank(self, query: str, *, candidate_topn: int = 100, topk: int = 10) -> list[Document]:
        docs = self.retriever.retrieve(query, topn=candidate_topn)
        if not docs:
            return []

        # Input shape is (None, list_size, num_features)
        expected_list_size = self.model.input_shape[1]
        num_features = self.model.input_shape[2]

        X = self.fe.extract_many(query, docs)
        num_docs = X.shape[0]

        X_padded = np.zeros((1, expected_list_size, num_features), dtype=np.float32)
        use_count = min(num_docs, expected_list_size)
        X_padded[0, :use_count, :] = X[:use_count, :]

        scores = self.model.predict(X_padded, verbose=0)[0]
        actual_scores = scores[:num_docs]
        order = np.argsort(-actual_scores)

        reranked = []
        for i in order[:topk]:
            if i >= len(docs):
                continue
            d = docs[int(i)]
            d.score = float(actual_scores[int(i)])
            reranked.append(d)
        return reranked

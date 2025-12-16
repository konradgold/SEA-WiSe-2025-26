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

    def rerank(self, query: str, *, candidate_topn: int = 200, topk: int = 10) -> list[Document]:
        docs = self.retriever.retrieve(query, topn=candidate_topn)
        if not docs:
            return []
        X = self.fe.extract_many(query, docs)  # [N, F]
        scores = self.model.predict(X[None, :, :], verbose=0)[0]  # [N]
        order = np.argsort(-scores)

        reranked = []
        for i in order[:topk]:
            d = docs[int(i)]
            d.score = float(scores[int(i)])
            reranked.append(d)
        return reranked





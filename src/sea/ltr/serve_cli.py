from __future__ import annotations

import argparse

from sea.ltr.serve_tfr import TFRReranker


def main() -> None:
    ap = argparse.ArgumentParser(description="Serve-time reranking: BM25 -> features -> TF-Ranking model -> top-10.")
    ap.add_argument("--model-path", type=str, required=True, help="Path to saved Keras model (model.keras).")
    ap.add_argument("--query", type=str, required=True)
    ap.add_argument("--candidate-topn", type=int, default=200)
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    rr = TFRReranker.load(model_path=args.model_path)
    docs = rr.rerank(args.query, candidate_topn=int(args.candidate_topn), topk=int(args.topk))
    for d in docs:
        d.pprint(verbose=False, loud=True)


if __name__ == "__main__":
    main()





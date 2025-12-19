from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import time

import numpy as np
import tensorflow as tf
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

from sea.ltr.bm25 import BM25Retriever
from sea.ltr.candidates import iter_qids, load_qrels_map, load_queries_map
from sea.ltr.features import FeatureExtractor
from sea.ltr.tfr_data import iter_listwise_samples
from sea.ltr.tfr_model import TFRConfig, build_tfr_scoring_model, compile_tfr_model
from sea.utils.config import Config


class WandbRankingTableCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_ds, queries, qrels, retriever, num_samples=5):
        super().__init__()
        self.val_ds = val_ds.take(1)  # Just take one batch
        self.queries = queries
        self.qrels = qrels
        self.retriever = retriever
        self.num_samples = num_samples

    def on_epoch_end(self, epoch, logs=None):
        table = wandb.Table(
            columns=["Epoch", "Query", "Predicted Ranks", "Top Doc Score", "Top Doc ID"]
        )

        # Get one batch of features and labels
        for features, labels in self.val_ds:
            scores = self.model.predict(features, verbose=0)
            for i in range(min(self.num_samples, len(scores))):
                # Simple visualization of the top doc in the list
                top_idx = np.argmax(scores[i])
                is_correct = labels[i][top_idx] == 1.0
                table.add_data(
                    epoch,
                    f"Sample {i}",
                    str(np.argsort(scores[i])[::-1]),
                    float(scores[i][top_idx]),
                    "Correct" if is_correct else "Incorrect",
                )
        wandb.log({"evaluation_samples": table})


def _tf_dataset_from_generator(
    gen_fn, *, list_size: int, num_features: int, batch_size: int
):
    output_signature = (
        tf.TensorSpec(shape=(list_size, num_features), dtype=tf.float32),
        tf.TensorSpec(shape=(list_size,), dtype=tf.float32),
    )

    def _gen():
        for s in gen_fn():
            yield s.features, s.labels

    ds = tf.data.Dataset.from_generator(_gen, output_signature=output_signature)
    ds = ds.shuffle(10_000, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a TensorFlow Ranking reranker on MS MARCO doc ranking data.")
    ap.add_argument("--queries", type=str, required=True)
    ap.add_argument("--qrels", type=str, required=True)
    ap.add_argument("--split-dir", type=str, required=True, help="Directory containing train_qids.txt/val_qids.txt/test_qids.txt")
    ap.add_argument("--candidate-topn", type=int, default=200)
    ap.add_argument("--list-size", type=int, default=9, help="1 positive + (list_size-1) negatives per instance.")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of queries for both train and val (for quick testing). 0 = no limit. Overrides --max-train-queries and --max-val-queries.",
    )
    ap.add_argument(
        "--max-train-queries",
        type=int,
        default=0,
        help="Limit number of training queries. 0 = no limit. Ignored if --limit is set.",
    )
    ap.add_argument(
        "--max-val-queries",
        type=int,
        default=2000,
        help="Limit number of validation queries. 0 = no limit. Ignored if --limit is set.",
    )
    ap.add_argument("--out-dir", type=str, default="artifacts/tfr_reranker")
    ap.add_argument(
        "--wandb-project", type=str, default="SEA-WiSe-2025-26", help="W&B project name"
    )
    ap.add_argument("--wandb-name", type=str, default=None, help="W&B run name")
    ap.add_argument(
        "--wandb-group", type=str, default="reranker", help="W&B group name"
    )
    ap.add_argument("--no-wandb", action="store_true", help="Disable W&B tracking")
    ap.add_argument(
        "--log-freq",
        type=int,
        default=50,
        help="Logging frequency in batches for W&B intermediate logging.",
    )
    args = ap.parse_args()

    cfg = Config(load=True)
    # Silence verbose output from the retriever during training to keep console clean
    cfg.SEARCH.VERBOSE_OUTPUT = False

    queries = load_queries_map(args.queries)
    qrels = load_qrels_map(args.qrels)

    split_dir = Path(args.split_dir)
    train_qids = list(iter_qids(split_dir / "train_qids.txt"))
    val_qids = list(iter_qids(split_dir / "val_qids.txt"))

    retriever = BM25Retriever.from_config(cfg)
    fe = FeatureExtractor.from_config(cfg)

    num_features = len(fe.features.names)
    list_size = int(args.list_size)

    # Use --limit if set, otherwise use individual args
    if args.limit > 0:
        max_train = int(args.limit)
        max_val = int(args.limit)
        print(f"Using --limit={args.limit}: max_train={max_train}, max_val={max_val}")
    else:
        max_train = None if args.max_train_queries == 0 else int(args.max_train_queries)
        max_val = None if args.max_val_queries == 0 else int(args.max_val_queries)
        if max_train is not None or max_val is not None:
            print(f"Query limits: max_train={max_train}, max_val={max_val}")

    def train_gen():
        return iter_listwise_samples(
            qids=train_qids,
            queries=queries,
            qrels=qrels,
            retriever=retriever,
            fe=fe,
            candidate_topn=int(args.candidate_topn),
            list_size=list_size,
            seed=int(args.seed),
            max_queries=max_train,
            description="Training data",
        )

    def val_gen():
        return iter_listwise_samples(
            qids=val_qids,
            queries=queries,
            qrels=qrels,
            retriever=retriever,
            fe=fe,
            candidate_topn=int(args.candidate_topn),
            list_size=list_size,
            seed=int(args.seed) + 1,
            max_queries=max_val,
            description="Validation data",
        )

    train_ds = _tf_dataset_from_generator(
        train_gen, list_size=list_size, num_features=num_features, batch_size=int(args.batch_size)
    )
    val_ds = _tf_dataset_from_generator(
        val_gen, list_size=list_size, num_features=num_features, batch_size=int(args.batch_size)
    )

    model = build_tfr_scoring_model(TFRConfig(list_size=list_size, num_features=num_features, learning_rate=float(args.lr)))
    model = compile_tfr_model(model, learning_rate=float(args.lr))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "feature_names.json").write_text(json.dumps(fe.features.names, indent=2) + "\n", encoding="utf-8")
    (out_dir / "train_args.json").write_text(json.dumps(vars(args), indent=2) + "\n", encoding="utf-8")

    callbacks = []
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            group=args.wandb_group,
            config=vars(args),
        )
        callbacks.append(WandbMetricsLogger(log_freq=args.log_freq))
        callbacks.append(
            WandbModelCheckpoint(
                filepath=str(out_dir / "best_model.keras"),
                monitor="val_mrr@10",
                mode="max",
                save_best_only=True,
            )
        )
        callbacks.append(
            WandbRankingTableCallback(val_ds, queries, qrels, retriever, num_samples=10)
        )

    t0 = time()
    hist = model.fit(
        train_ds, validation_data=val_ds, epochs=int(args.epochs), callbacks=callbacks
    )
    elapsed = time() - t0

    # Save model
    model_path = out_dir / "model.keras"
    model.save(model_path)

    # Save history for plotting
    history_path = out_dir / "history.json"
    clean_hist = {k: [float(x) for x in v] for k, v in hist.history.items()}
    history_path.write_text(json.dumps(clean_hist, indent=2) + "\n", encoding="utf-8")

    print(f"Saved model to {model_path}")
    print(f"Saved history to {history_path}")
    print(f"Training time: {elapsed:.1f}s")

    # Print final summary of best metrics
    print("\n" + "=" * 30)
    print("FINAL TRAINING SUMMARY")
    print("=" * 30)
    for metric, values in clean_hist.items():
        if metric.startswith("val_"):
            best_val = max(values) if not metric.endswith("loss") else min(values)
            print(f"{metric:20}: {best_val:.4f}")
    print("=" * 30)


if __name__ == "__main__":
    main()

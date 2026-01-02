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
    ds = ds.shuffle(1_000, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def _tf_dataset_from_npz(path: str, *, batch_size: int, shuffle: bool = True):
    print(f"Loading pre-computed data from {path}...")
    data = np.load(path)
    features = data["features"]
    labels = data["labels"]
    print(f"Loaded {len(features)} samples.")

    ds = tf.data.Dataset.from_tensor_slices((features, labels))
    if shuffle:
        ds = ds.shuffle(len(features), reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def main() -> None:
    cfg = Config(load=True)
    cfg.SEARCH.VERBOSE_OUTPUT = False

    ap = argparse.ArgumentParser(description="Train a TensorFlow Ranking reranker on MS MARCO doc ranking data.")
    ap.add_argument("--queries", type=str, default=getattr(cfg.LTR, "QUERIES", None))
    ap.add_argument("--qrels", type=str, default=getattr(cfg.LTR, "QRELS", None))
    ap.add_argument(
        "--split-dir", type=str, default=getattr(cfg.LTR, "SPLIT_DIR", None)
    )
    ap.add_argument(
        "--candidate-topn", type=int, default=getattr(cfg.LTR, "CANDIDATE_TOPN", 200)
    )
    ap.add_argument("--list-size", type=int, default=getattr(cfg.LTR, "LIST_SIZE", 100))
    ap.add_argument("--epochs", type=int, default=getattr(cfg.LTR, "EPOCHS", 3))
    ap.add_argument(
        "--batch-size", type=int, default=getattr(cfg.LTR, "BATCH_SIZE", 64)
    )
    ap.add_argument("--lr", type=float, default=getattr(cfg.LTR, "LEARNING_RATE", 1e-3))

    scheduler_cfg = getattr(cfg.LTR, "SCHEDULER", None)
    ap.add_argument(
        "--lr-scheduler", type=str, default=getattr(scheduler_cfg, "TYPE", "none")
    )
    ap.add_argument(
        "--lr-decay-steps",
        type=int,
        default=getattr(scheduler_cfg, "DECAY_STEPS", 10000),
    )
    ap.add_argument(
        "--lr-decay-rate", type=float, default=getattr(scheduler_cfg, "DECAY_RATE", 0.9)
    )
    ap.add_argument(
        "--lr-alpha", type=float, default=getattr(scheduler_cfg, "ALPHA", 0.1)
    )

    ap.add_argument("--seed", type=int, default=getattr(cfg.LTR, "SEED", 42))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-train-queries", type=int, default=0)
    ap.add_argument("--max-val-queries", type=int, default=2000)
    ap.add_argument("--out-dir", type=str, default="artifacts/tfr_reranker")
    ap.add_argument(
        "--train-cache", type=str, default=getattr(cfg.LTR, "TRAIN_CACHE", None)
    )
    ap.add_argument(
        "--val-cache", type=str, default=getattr(cfg.LTR, "VAL_CACHE", None)
    )

    wandb_cfg = getattr(cfg.LTR, "WANDB", None)
    ap.add_argument(
        "--wandb-project",
        type=str,
        default=getattr(wandb_cfg, "PROJECT", "SEA-WiSe-2025-26"),
    )
    ap.add_argument("--wandb-name", type=str, default=None)
    ap.add_argument(
        "--wandb-group", type=str, default=getattr(wandb_cfg, "GROUP", "reranker")
    )
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument(
        "--wandb-log-model",
        action="store_true",
        default=getattr(wandb_cfg, "LOG_MODEL", False),
    )
    ap.add_argument("--log-freq", type=int, default=5)
    args = ap.parse_args()

    if not (args.train_cache and args.val_cache):
        if not (args.queries and args.qrels and args.split_dir):
            ap.error(
                "Must provide --queries, --qrels, and --split-dir if caches are not provided."
            )

    queries = load_queries_map(args.queries) if args.queries else {}
    qrels = load_qrels_map(args.qrels) if args.qrels else {}

    train_qids = []
    val_qids = []
    if args.split_dir:
        split_dir = Path(args.split_dir)
        train_qids = list(iter_qids(split_dir / "train_qids.txt"))
        val_qids = list(iter_qids(split_dir / "val_qids.txt"))

    retriever = BM25Retriever.from_config(cfg)
    fe = FeatureExtractor.from_config(cfg)

    if args.train_cache:
        with np.load(args.train_cache) as data:
            num_features = data["features"].shape[2]
            list_size = data["features"].shape[1]
    else:
        num_features = len(fe.features.names)
        list_size = int(args.list_size)

    max_train = (
        int(args.limit)
        if args.limit > 0
        else (None if args.max_train_queries == 0 else int(args.max_train_queries))
    )
    max_val = (
        int(args.limit)
        if args.limit > 0
        else (None if args.max_val_queries == 0 else int(args.max_val_queries))
    )

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

    if args.train_cache:
        train_ds = _tf_dataset_from_npz(
            args.train_cache, batch_size=int(args.batch_size), shuffle=True
        )
    else:
        train_ds = _tf_dataset_from_generator(
            train_gen,
            list_size=list_size,
            num_features=num_features,
            batch_size=int(args.batch_size),
        )

    if args.val_cache:
        val_ds = _tf_dataset_from_npz(
            args.val_cache, batch_size=int(args.batch_size), shuffle=False
        )
    else:
        val_ds = _tf_dataset_from_generator(
            val_gen,
            list_size=list_size,
            num_features=num_features,
            batch_size=int(args.batch_size),
        )

    model_params = getattr(cfg.LTR, "MODEL", None)
    hidden_units = tuple(getattr(model_params, "HIDDEN_UNITS", (128, 64, 32)))
    dropout = getattr(model_params, "DROPOUT", 0.1)
    use_attention = getattr(model_params, "USE_ATTENTION", True)

    lr_or_scheduler = float(args.lr)
    if args.lr_scheduler == "exponential":
        lr_or_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=args.lr,
            decay_steps=args.lr_decay_steps,
            decay_rate=args.lr_decay_rate,
        )
    elif args.lr_scheduler == "cosine":
        lr_or_scheduler = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=args.lr,
            decay_steps=args.lr_decay_steps,
            alpha=args.lr_alpha,
        )

    model_cfg = TFRConfig(
        list_size=list_size,
        num_features=num_features,
        learning_rate=lr_or_scheduler,
        hidden_units=hidden_units,
        dropout=dropout,
        use_attention=use_attention,
    )
    model = build_tfr_scoring_model(model_cfg)

    adaptation_data = train_ds.take(20).map(lambda x, y: x)
    model.get_layer("normalization").adapt(adaptation_data)

    model = compile_tfr_model(model, learning_rate=lr_or_scheduler)

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
        checkpoint_cls = (
            WandbModelCheckpoint
            if args.wandb_log_model
            else tf.keras.callbacks.ModelCheckpoint
        )
        callbacks.append(
            checkpoint_cls(
                filepath=str(out_dir / "best_model.keras"),
                monitor="val_mrr@10",
                mode="max",
                save_best_only=True,
            )
        )

    steps_per_epoch = None
    if not args.train_cache and args.limit > 0:
        steps_per_epoch = max(1, int(args.limit * 0.8) // int(args.batch_size))

    t0 = time()
    hist = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=int(args.epochs),
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch,
        validation_steps=steps_per_epoch // 4 if steps_per_epoch else None,
    )
    elapsed = time() - t0

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

    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()

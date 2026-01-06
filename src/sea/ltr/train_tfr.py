from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import time

import hydra
import numpy as np
import tensorflow as tf
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

from sea.ltr.bm25 import BM25Retriever
from sea.ltr.candidates import iter_qids, load_qrels_map, load_queries_map
from sea.ltr.features import FeatureExtractor
from sea.ltr.tfr_data import iter_listwise_samples
from sea.ltr.tfr_model import TFRConfig, build_tfr_scoring_model, compile_tfr_model
from omegaconf import DictConfig


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

@hydra.main(version_base=None, config_path="../../configs", config_name="train_ltr")
def main(cfg: DictConfig) -> None:
    cfg.SEARCH.VERBOSE_OUTPUT = False

    ap = argparse.ArgumentParser(description="Train a TensorFlow Ranking reranker on MS MARCO doc ranking data.")
    # TODO: Solve using hydra?
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument(
        "--wandb-run-name",
        type=str,
        default="tfr_reranker",
        help="Name of the Weights & Biases run.",
    )
    args = ap.parse_args()

    if not (cfg.LTR.TRAIN_CACHE and cfg.LTR.VAL_CACHE):
        if not (cfg.LTR.QUERIES and cfg.LTR.QRELS and cfg.LTR.SPLIT_DIR):
            ap.error(
                "Must provide --queries, --qrels, and --split-dir if caches are not provided."
            )

    queries = load_queries_map(cfg.LTR.QUERIES) if cfg.LTR.QUERIES else {}
    qrels = load_qrels_map(cfg.LTR.QRELS) if cfg.LTR.QRELS else {}

    train_qids = []
    val_qids = []
    if cfg.LTR.SPLIT_DIR:
        split_dir = Path(cfg.LTR.SPLIT_DIR)
        train_qids = list(iter_qids(split_dir / "train_qids.txt"))
        val_qids = list(iter_qids(split_dir / "val_qids.txt"))

    retriever = BM25Retriever.from_config(cfg)
    fe = FeatureExtractor.from_config(cfg)

    if cfg.LTR.TRAIN_CACHE:
        with np.load(cfg.LTR.TRAIN_CACHE) as data:
            num_features = data["features"].shape[2]
            list_size = data["features"].shape[1]
    else:
        num_features = len(fe.features.names)
        list_size = int(cfg.LTR.LIST_SIZE)

    max_train = (
        int(cfg.limit)
        if cfg.limit > 0
        else (None if cfg.max_train_queries == 0 else int(cfg.max_train_queries))
    )
    max_val = (
        int(cfg.limit)
        if cfg.limit > 0
        else (None if cfg.max_val_queries == 0 else int(cfg.max_val_queries))
    )

    def train_gen():
        return iter_listwise_samples(
            qids=train_qids,
            queries=queries,
            qrels=qrels,
            retriever=retriever,
            fe=fe,
            candidate_topn=int(cfg.LTR.CANDIDATE_TOPN),
            list_size=list_size,
            seed=int(cfg.LTR.SEED),
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
            candidate_topn=int(cfg.LTR.CANDIDATE_TOPN),
            list_size=list_size,
            seed=int(cfg.LTR.SEED) + 1,
            max_queries=max_val,
            description="Validation data",
        )

    if cfg.LTR.TRAIN_CACHE:
        train_ds = _tf_dataset_from_npz(
            cfg.LTR.TRAIN_CACHE, batch_size=int(cfg.LTR.BATCH_SIZE), shuffle=True
        )
    else:
        train_ds = _tf_dataset_from_generator(
            train_gen,
            list_size=list_size,
            num_features=num_features,
            batch_size=int(cfg.LTR.BATCH_SIZE),
        )

    if cfg.LTR.VAL_CACHE:
        val_ds = _tf_dataset_from_npz(
            cfg.LTR.VAL_CACHE, batch_size=int(cfg.LTR.BATCH_SIZE), shuffle=False
        )
    else:
        val_ds = _tf_dataset_from_generator(
            val_gen,
            list_size=list_size,
            num_features=num_features,
            batch_size=int(cfg.LTR.BATCH_SIZE),
        )

    model_params = getattr(cfg.LTR, "MODEL", None)
    hidden_units = tuple(getattr(model_params, "HIDDEN_UNITS", (128, 64, 32)))
    dropout = getattr(model_params, "DROPOUT", 0.1)
    use_attention = getattr(model_params, "USE_ATTENTION", True)

    lr_or_scheduler = float(cfg.LTR.LEARNING_RATE)
    if cfg.LTR.SCHEDULER.TYPE == "exponential":
        lr_or_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=cfg.LTR.LEARNING_RATE,
            decay_steps=cfg.LTR.SCHEDULER.DECAY_STEPS,
            decay_rate=cfg.LTR.SCHEDULER.DECAY_RATE,
        )
    elif cfg.LTR.SCHEDULER.TYPE == "cosine":
        lr_or_scheduler = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=cfg.LTR.LEARNING_RATE,
            decay_steps=cfg.LTR.SCHEDULER.DECAY_STEPS,
            alpha=cfg.LTR.SCHEDULER.ALPHA,
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

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "feature_names.json").write_text(json.dumps(fe.features.names, indent=2) + "\n", encoding="utf-8")
    (out_dir / "train_args.json").write_text(json.dumps(vars(cfg), indent=2) + "\n", encoding="utf-8")

    callbacks = []
    if not args.no_wandb:
        wandb.init(
            project=cfg.LTR.WANDB.PROJECT,
            name=args.wandb_run_name,
            group=cfg.LTR.WANDB.GROUP,
            config=vars(cfg),
        )
        callbacks.append(WandbMetricsLogger(log_freq=cfg.log_freq))
        checkpoint_cls = (
            WandbModelCheckpoint
            if cfg.LTR.WANDB.LOG_MODEL
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
    if not cfg.LTR.TRAIN_CACHE and cfg.limit > 0:
        steps_per_epoch = max(1, int(cfg.limit * 0.8) // int(cfg.LTR.BATCH_SIZE))

    t0 = time()
    hist = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=int(cfg.LTR.EPOCHS),
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

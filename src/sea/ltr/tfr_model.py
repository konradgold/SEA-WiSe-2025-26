from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TFRConfig:
    list_size: int
    num_features: int
    hidden_units: tuple[int, ...] = (64, 32)
    dropout: float = 0.1
    learning_rate: float = 1e-3


def build_tfr_scoring_model(cfg: TFRConfig):
    """
    Builds a Keras model that maps:
      features: [batch, list_size, num_features] -> scores: [batch, list_size]

    We keep it intentionally small; ranking quality > score calibration.
    """
    import tensorflow as tf

    inputs = tf.keras.Input(shape=(cfg.list_size, cfg.num_features), dtype=tf.float32, name="features")
    x = inputs
    for i, h in enumerate(cfg.hidden_units):
        x = tf.keras.layers.Dense(h, activation="relu", name=f"dense_{i}")(x)
        if cfg.dropout and cfg.dropout > 0:
            x = tf.keras.layers.Dropout(cfg.dropout, name=f"dropout_{i}")(x)
    x = tf.keras.layers.Dense(1, activation=None, name="score")(x)  # [B, L, 1]
    scores = tf.squeeze(x, axis=-1, name="scores")  # [B, L]
    return tf.keras.Model(inputs=inputs, outputs=scores, name="tfr_reranker")


def compile_tfr_model(model, *, learning_rate: float = 1e-3):
    """
    Pairwise logistic ranking loss over lists.
    """
    import tensorflow as tf
    import tensorflow_ranking as tfr
    import sys

    # Use legacy tensorflow optimizer for M1 Mac
    if sys.platform == "darwin":
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    loss = tfr.keras.losses.PairwiseLogisticLoss()
    metrics = [
        tfr.keras.metrics.MRRMetric(name="mrr@10", topn=10),
        tfr.keras.metrics.NDCGMetric(name="ndcg@10", topn=10),
        tfr.keras.metrics.RecallMetric(name="recall@10", topn=10),
    ]
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
    )
    return model

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Union, Any


@dataclass(frozen=True)
class TFRConfig:
    list_size: int
    num_features: int
    hidden_units: tuple[int, ...] = (128, 64, 32)
    dropout: float = 0.1
    learning_rate: Union[float, Any] = 1e-3
    use_attention: bool = True


def build_tfr_scoring_model(cfg: TFRConfig):
    """
    Builds a Keras model that maps:
      features: [batch, list_size, num_features] -> scores: [batch, list_size]
    """
    import tensorflow as tf

    inputs = tf.keras.Input(shape=(cfg.list_size, cfg.num_features), dtype=tf.float32, name="features")

    x = tf.keras.layers.Normalization(axis=-1, name="normalization")(inputs)
    x = tf.keras.layers.Dense(cfg.hidden_units[0], activation="relu", name="dense_0")(x)

    if cfg.use_attention:
        attn_output = tf.keras.layers.MultiHeadAttention(
            num_heads=4, key_dim=cfg.hidden_units[0] // 4, name="self_attention"
        )(x, x)
        x = tf.keras.layers.Add()([x, attn_output])  # Residual connection
        x = tf.keras.layers.LayerNormalization()(x)

    for i, h in enumerate(cfg.hidden_units[1:]):
        x = tf.keras.layers.Dense(h, activation="relu", name=f"dense_{i+1}")(x)
        if cfg.dropout > 0:
            x = tf.keras.layers.Dropout(cfg.dropout, name=f"dropout_{i+1}")(x)

    x = tf.keras.layers.Dense(1, activation=None, name="score")(x)
    scores = tf.squeeze(x, axis=-1, name="scores")
    return tf.keras.Model(inputs=inputs, outputs=scores, name="tfr_reranker")


def compile_tfr_model(model, *, learning_rate: Union[float, Any] = 1e-3):
    """
    ApproxNDCGLoss is a listwise loss that directly optimizes a smooth version of NDCG.
    """
    import tensorflow as tf
    import tensorflow_ranking as tfr

    # Use legacy tensorflow optimizer for M1 Mac
    if sys.platform == "darwin":
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss=tfr.keras.losses.ApproxNDCGLoss(),
        metrics=[
            tfr.keras.metrics.MRRMetric(name="mrr@1", topn=1),
            tfr.keras.metrics.MRRMetric(name="mrr@10", topn=10),
            tfr.keras.metrics.NDCGMetric(name="ndcg@10", topn=10),
            tfr.keras.metrics.RecallMetric(name="recall@1", topn=1),
        ],
    )
    return model

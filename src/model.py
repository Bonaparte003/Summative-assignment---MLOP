from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0


def build_efficientnet_binary_classifier(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    train_base: bool = False,
    unfreeze_last_n: int = 0,
) -> tf.keras.Model:
    """
    Binary classifier:
      output 0..1 probability (sigmoid)

    If train_base is True, the whole EfficientNet backbone is trainable.
    Else if unfreeze_last_n > 0, only the last N backbone layers are trainable
    (matches the intro notebook fine-tuning recipe, typically N=30).
    Otherwise the backbone is frozen (head-only training).
    """
    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )

    if train_base:
        for layer in base_model.layers:
            layer.trainable = True
        base_training = True
    elif unfreeze_last_n > 0:
        base_model.trainable = True
        for layer in base_model.layers[:-unfreeze_last_n]:
            layer.trainable = False
        for layer in base_model.layers[-unfreeze_last_n:]:
            layer.trainable = True
        base_training = True
    else:
        base_model.trainable = False
        for layer in base_model.layers:
            layer.trainable = False
        base_training = False

    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=base_training)
    x = layers.GlobalAveragePooling2D()(x)
    # Hidden head matches the original notebook EfficientNet experiments.
    x = layers.Dense(128, activation="relu", dtype="float32")(x)
    x = layers.Dropout(0.2)(x)

    # Keep output dtype stable even when mixed precision is enabled.
    outputs = layers.Dense(1, activation="sigmoid", dtype="float32")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="efficientnet_binary")


def compile_for_binary_classification(
    model: tf.keras.Model,
    learning_rate: float = 1e-4,
) -> None:
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )


import argparse
import json
import os
from datetime import datetime
from typing import Optional

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .model import build_efficientnet_binary_classifier, compile_for_binary_classification
from .preprocessing import (
    load_images_affectnet_equal_classes,
    load_images_and_labels_from_binary_folders,
    preprocess_efficientnet_rgb_uint8,
    train_val_test_split_balanced,
)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _save_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _evaluate_binary(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(np.int64)

    # Avoid crashes when one class is missing in a split.
    results = {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, average="binary", zero_division=0)
        ),
        "recall": float(recall_score(y_true, y_pred, average="binary", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
        "roc_auc": None,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
    }

    try:
        results["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        results["roc_auc"] = None

    return results


def train(
    data_dir: str,
    uploads_dir: Optional[str],
    output_dir: str,
    epochs: int = 10,
    batch_size: int = 16,
    image_size: int = 224,
    limit_per_emotion: Optional[int] = 400,
    train_base: bool = False,
    unfreeze_last_n: int = 0,
    learning_rate: float = 1e-4,
    seed: int = 42,
) -> dict:
    """
    Trains the binary EfficientNet model and saves:
      - output_dir/models/idetect_classifier.keras
      - output_dir/reports/metrics.json
    Returns metrics summary.
    """
    tf.keras.utils.set_random_seed(seed)

    
    try:
        policy = tf.keras.mixed_precision.Policy("mixed_float16")
        tf.keras.mixed_precision.set_global_policy(policy)
    except Exception:
        pass

    X_em, y_em = load_images_affectnet_equal_classes(
        data_dir=data_dir,
        image_size=(image_size, image_size),
        limit_per_emotion=limit_per_emotion,
        seed=seed,
    )

    if uploads_dir:
        X_up, y_up = load_images_and_labels_from_binary_folders(
            uploads_dir=uploads_dir,
            image_size=(image_size, image_size),
            limit_per_label=None,
        )
        if len(X_up) > 0:
            X_all = np.concatenate([X_em, X_up], axis=0)
            y_all = np.concatenate([y_em, y_up], axis=0)
        else:
            X_all, y_all = X_em, y_em
    else:
        X_all, y_all = X_em, y_em

    X_all = preprocess_efficientnet_rgb_uint8(X_all)

    split = train_val_test_split_balanced(X_all, y_all, seed=seed)
    (X_train, y_train) = split.train
    (X_val, y_val) = split.val
    (X_test, y_test) = split.test

    def make_ds(X: np.ndarray, y: np.ndarray, training: bool) -> tf.data.Dataset:
        ds = tf.data.Dataset.from_tensor_slices((X, y))
        if training:
            ds = ds.shuffle(buffer_size=min(2048, len(X)), seed=seed)
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    train_ds = make_ds(X_train, y_train, training=True)
    val_ds = make_ds(X_val, y_val, training=False)
    test_ds = make_ds(X_test, y_test, training=False)

    model = build_efficientnet_binary_classifier(
        input_shape=(image_size, image_size, 3),
        train_base=train_base,
        unfreeze_last_n=0 if train_base else unfreeze_last_n,
    )
    compile_for_binary_classification(model, learning_rate=learning_rate)

    _ensure_dir(os.path.join(output_dir, "models"))
    _ensure_dir(os.path.join(output_dir, "reports"))

    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    ckpt_path = os.path.join(output_dir, "models", "idetect_classifier.keras")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_path,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate on test set
    y_prob = model.predict(test_ds, verbose=0).ravel()
    metrics = _evaluate_binary(y_test, y_prob, threshold=0.5)

    metrics_payload = {
        "run_id": run_id,
        "data_dir": data_dir,
        "uploads_dir": uploads_dir,
        "image_size": image_size,
        "batch_size": batch_size,
        "epochs": epochs,
        "limit_per_emotion": limit_per_emotion,
        "train_base": train_base,
        "unfreeze_last_n": unfreeze_last_n if not train_base else None,
        "learning_rate": learning_rate,
        "split_sizes": {
            "train": int(len(X_train)),
            "val": int(len(X_val)),
            "test": int(len(X_test)),
        },
        "metrics": metrics,
        "history": history.history,
    }

    _save_json(os.path.join(output_dir, "reports", "metrics.json"), metrics_payload)

    _save_json(
        os.path.join(output_dir, "reports", "model_meta.json"),
        {
            "run_id": run_id,
            "saved_model_path": ckpt_path,
            "threshold": 0.5,
        },
    )
    return metrics_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and export a binary EfficientNet model.")
    parser.add_argument("--data_dir", required=True, help="AffectNet root folder")
    parser.add_argument("--uploads_dir", default=None, help="Optional uploads root with 0/ and 1/ subfolders")
    parser.add_argument("--output_dir", default=".", help="Where to save models/ and reports/")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--limit_per_emotion", type=int, default=400)
    parser.add_argument(
        "--train_base",
        action="store_true",
        help="Fine-tune the entire EfficientNet base (overrides --unfreeze_last_n)",
    )
    parser.add_argument(
        "--unfreeze_last_n",
        type=int,
        default=0,
        help="Fine-tune only the last N backbone layers (e.g. 20 to match the Colab notebook)",
    )
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        uploads_dir=args.uploads_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        limit_per_emotion=args.limit_per_emotion,
        train_base=args.train_base,
        unfreeze_last_n=args.unfreeze_last_n,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()


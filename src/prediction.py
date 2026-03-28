import io
import json
import os
import tempfile
import zipfile
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess_input


LABELS: Dict[int, str] = {
    1: "approachable",
    0: "not_approachable",
}


def _strip_quantization_config(obj: Any) -> None:
    """Remove keys newer Keras versions add that older local Keras rejects on load."""
    if isinstance(obj, dict):
        obj.pop("quantization_config", None)
        for v in obj.values():
            _strip_quantization_config(v)
    elif isinstance(obj, list):
        for item in obj:
            _strip_quantization_config(item)


def load_keras_model_compat(model_path: str, compile: bool = False):
    """
    Load a .keras model; if the archive was saved with a newer Keras that embeds
    ``quantization_config`` on layers, strip it so older TensorFlow/Keras can load.
    """
    path = os.path.abspath(model_path)
    try:
        return tf.keras.models.load_model(path, compile=compile)
    except (TypeError, ValueError) as e:
        err = str(e).lower()
        if "quantization_config" not in err and "could not be deserialized" not in err:
            raise
    if not path.lower().endswith(".keras"):
        raise

    buf = io.BytesIO()
    with zipfile.ZipFile(path, "r") as zin:
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                data = zin.read(item.filename)
                if item.filename.endswith(".json"):
                    try:
                        cfg = json.loads(data.decode("utf-8"))
                        _strip_quantization_config(cfg)
                        data = json.dumps(cfg).encode("utf-8")
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        pass
                zout.writestr(item, data)
    buf.seek(0)
    fd, tmp_path = tempfile.mkstemp(suffix=".keras")
    os.close(fd)
    try:
        with open(tmp_path, "wb") as f:
            f.write(buf.getvalue())
        return tf.keras.models.load_model(tmp_path, compile=compile)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def preprocess_image_bytes(
    image_bytes: bytes,
    image_size: Tuple[int, int] = (224, 224),
) -> np.ndarray:
    """
    Decodes the image, resizes to image_size, and applies EfficientNet
    keras.applications preprocessing (must match training / notebook).
    """
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image. Please upload a valid image file.")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, image_size)
    x = eff_preprocess_input(np.expand_dims(img.astype(np.uint8), axis=0))
    return x


class Predictor:
    def __init__(self, model_path: str, image_size: Tuple[int, int] = (224, 224)):
        self.image_size = image_size
        self.model = load_keras_model_compat(model_path, compile=False)

    def predict_proba(self, image_bytes: bytes) -> float:
        X = preprocess_image_bytes(image_bytes, image_size=self.image_size)
        y_prob = self.model.predict(X, verbose=0).ravel()[0]
        return float(y_prob)

    def predict(
        self,
        image_bytes: bytes,
        threshold: float = 0.5,
    ) -> Dict:
        prob = self.predict_proba(image_bytes=image_bytes)
        pred_label = 1 if prob >= threshold else 0
        return {
            "probability": prob,
            "threshold": threshold,
            "predicted_label": pred_label,
            "predicted_class": LABELS[pred_label],
        }


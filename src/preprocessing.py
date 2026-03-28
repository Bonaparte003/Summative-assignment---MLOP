import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from sklearn.model_selection import train_test_split


LABEL_MAPPING: Dict[str, int] = {
    "happy": 1,  # Approachable
    "neutral": 1,  # Approachable
    "anger": 0,  # Not approachable
    "contempt": 0,  # Not approachable
}


@dataclass(frozen=True)
class Split:
    train: Tuple[np.ndarray, np.ndarray]
    val: Tuple[np.ndarray, np.ndarray]
    test: Tuple[np.ndarray, np.ndarray]


def _list_image_files(folder: str) -> List[str]:
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    try:
        return [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(exts)
        ]
    except FileNotFoundError:
        return []


def load_images_and_labels_from_emotion_folders(
    data_dir: str,
    image_size: Tuple[int, int] = (224, 224),
    limit_per_emotion: Optional[int] = 400,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads images from AffectNet-like folder structure:
      data_dir/{happy,neutral,anger,contempt}/*.jpg|png
    Maps emotion folders to binary labels via LABEL_MAPPING.
    """
    image_h, image_w = image_size[1], image_size[0]
    images: List[np.ndarray] = []
    labels: List[int] = []

    emotion_folders = ["happy", "neutral", "anger", "contempt"]
    for emotion in emotion_folders:
        emotion_dir = os.path.join(data_dir, emotion)
        binary_label = LABEL_MAPPING.get(emotion, -1)
        if binary_label < 0:
            continue

        files = _list_image_files(emotion_dir)
        if not files:
            continue

        if limit_per_emotion is not None:
            files = files[:limit_per_emotion]

        for fp in files:
            img = cv2.imread(fp)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, image_size)
            # Ensure we keep consistent dtype (model code may normalize)
            images.append(img.astype(np.uint8))
            labels.append(binary_label)

    if not images:
        raise RuntimeError(
            f"No images found in '{data_dir}'. Expected subfolders: {emotion_folders}"
        )

    X = np.stack(images, axis=0).reshape(-1, image_size[1], image_size[0], 3)
    y = np.asarray(labels, dtype=np.int64)
    return X, y


def load_images_affectnet_equal_classes(
    data_dir: str,
    image_size: Tuple[int, int] = (224, 224),
    limit_per_emotion: Optional[int] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Same sampling policy as ``notebook/iDetect_Project_Summative_machine_learning_pipeline.ipynb``:
    equal counts from happy/neutral, equal from anger/contempt, then N = min(pair caps) so
    class 0 and class 1 each have exactly 2*N images.
    """
    rng = np.random.default_rng(seed)
    emotion_order = ["happy", "neutral", "anger", "contempt"]
    by_emotion: Dict[str, List[np.ndarray]] = {e: [] for e in emotion_order}

    for emotion in emotion_order:
        emotion_dir = os.path.join(data_dir, emotion)
        files = _list_image_files(emotion_dir)
        rng.shuffle(files)
        if limit_per_emotion is not None:
            files = files[:limit_per_emotion]
        for fp in files:
            img = cv2.imread(fp)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, image_size)
            by_emotion[emotion].append(img.astype(np.uint8))

    n_h, n_n = len(by_emotion["happy"]), len(by_emotion["neutral"])
    n_a, n_c = len(by_emotion["anger"]), len(by_emotion["contempt"])
    pair1 = min(n_h, n_n)
    pair0 = min(n_a, n_c)
    if pair1 == 0 or pair0 == 0:
        raise RuntimeError(
            f"Need images in all four folders under '{data_dir}'. "
            f"Counts: happy={n_h} neutral={n_n} anger={n_a} contempt={n_c}"
        )
    n_take = min(pair1, pair0)
    imgs0 = by_emotion["anger"][:n_take] + by_emotion["contempt"][:n_take]
    imgs1 = by_emotion["happy"][:n_take] + by_emotion["neutral"][:n_take]
    x_list = imgs0 + imgs1
    y_arr = np.asarray([0] * (2 * n_take) + [1] * (2 * n_take), dtype=np.int64)
    perm = rng.permutation(len(y_arr))
    X = np.stack([x_list[i] for i in perm])
    y = y_arr[perm]
    return X.reshape(-1, image_size[1], image_size[0], 3), y


def load_images_and_labels_from_binary_folders(
    uploads_dir: str,
    image_size: Tuple[int, int] = (224, 224),
    limit_per_label: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads images from:
      uploads_dir/0/*.jpg|png
      uploads_dir/1/*.jpg|png
    Labels are the folder names (0 or 1).
    """
    images: List[np.ndarray] = []
    labels: List[int] = []

    for label_str in ["0", "1"]:
        label = int(label_str)
        label_dir = os.path.join(uploads_dir, label_str)
        files = _list_image_files(label_dir)
        if not files:
            continue
        if limit_per_label is not None:
            files = files[:limit_per_label]

        for fp in files:
            img = cv2.imread(fp)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, image_size)
            images.append(img.astype(np.uint8))
            labels.append(label)

    if not images:
        return np.empty((0, image_size[1], image_size[0], 3), dtype=np.uint8), np.empty(
            (0,), dtype=np.int64
        )

    X = np.stack(images, axis=0).reshape(-1, image_size[1], image_size[0], 3)
    y = np.asarray(labels, dtype=np.int64)
    return X, y


def train_val_test_split_balanced(
    X: np.ndarray,
    y: np.ndarray,
    train_size: float = 0.6,
    val_size: float = 0.2,
    test_size: float = 0.2,
    seed: int = 42,
) -> Split:
    """
    Stratified split that matches the notebook proportions (60/20/20 by default).
    """
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("train_size + val_size + test_size must equal 1.0")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=(val_size + test_size),
        random_state=seed,
        stratify=y,
    )

    # Split temp into val/test
    val_fraction_of_temp = val_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=(1.0 - val_fraction_of_temp),
        random_state=seed,
        stratify=y_temp,
    )

    return Split(
        train=(X_train, y_train),
        val=(X_val, y_val),
        test=(X_test, y_test),
    )


def preprocess_efficientnet_rgb_uint8(X_uint8: np.ndarray) -> np.ndarray:
    """
    RGB uint8 images (N, H, W, 3) in [0, 255] -> Keras EfficientNet preprocess_input
    (same as the original iDetect notebook DL experiments). Required for correct
    transfer learning from ImageNet weights.
    """
    from tensorflow.keras.applications.efficientnet import preprocess_input

    if X_uint8.dtype != np.uint8:
        X_uint8 = np.clip(X_uint8, 0, 255).astype(np.uint8)
    return preprocess_input(np.copy(X_uint8))


def normalize_images_uint8_to_float01(X_uint8: np.ndarray) -> np.ndarray:
    """
    Convert uint8 [0..255] images into float32 [0..1].
    """
    if X_uint8.dtype != np.uint8:
        X_uint8 = X_uint8.astype(np.uint8)
    return (X_uint8.astype(np.float32) / 255.0).astype(np.float32)


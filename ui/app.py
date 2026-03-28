import os
import time
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import requests
import streamlit as st


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_API_URL = os.getenv("API_URL", "http://localhost:8000")
DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR / "AffectNet")))

LABEL_MAPPING = {
    "happy": 1,
    "neutral": 1,
    "anger": 0,
    "contempt": 0,
}


st.set_page_config(page_title="iDetect UI", layout="wide")


def load_dataset_counts() -> Dict[int, int]:
    counts = {0: 0, 1: 0}
    for emotion, lbl in LABEL_MAPPING.items():
        emotion_dir = DATA_DIR / emotion
        if not emotion_dir.exists():
            continue
        files = [f for f in emotion_dir.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        counts[lbl] += len(files)
    return counts


def compute_three_feature_hists(max_images_per_emotion: int = 50, image_size: int = 224):
    """
    Simple numeric features for interpretability:
      1) brightness (mean pixel intensity)
      2) edge density (Canny edge pixels / total pixels)
      3) gradient magnitude (Sobel magnitude mean)
    """
    brightness = {0: [], 1: []}
    edge_density = {0: [], 1: []}
    gradient_mag = {0: [], 1: []}

    for emotion, lbl in LABEL_MAPPING.items():
        emotion_dir = DATA_DIR / emotion
        if not emotion_dir.exists():
            continue
        img_files = [f for f in emotion_dir.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        img_files = img_files[:max_images_per_emotion]

        for fp in img_files:
            img_bgr = cv2.imread(str(fp))
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_rgb = cv2.resize(img_rgb, (image_size, image_size))
            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

            brightness[lbl].append(float(gray.mean()))
            edges = cv2.Canny(gray, 100, 200)
            edge_density[lbl].append(float(np.mean(edges > 0)))
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            mag = np.sqrt(gx * gx + gy * gy)
            gradient_mag[lbl].append(float(mag.mean()))

    def to_hist_data(arrs: List[float]) -> np.ndarray:
        if not arrs:
            return np.array([0.0])
        return np.asarray(arrs, dtype=np.float32)

    return {
        "brightness": {0: to_hist_data(brightness[0]), 1: to_hist_data(brightness[1])},
        "edge_density": {0: to_hist_data(edge_density[0]), 1: to_hist_data(edge_density[1])},
        "gradient_mag": {0: to_hist_data(gradient_mag[0]), 1: to_hist_data(gradient_mag[1])},
    }


def plot_hist(ax, arr0: np.ndarray, arr1: np.ndarray, title: str, xlabel: str):
    ax.hist(arr0, bins=30, alpha=0.6, label="not_approachable (0)")
    ax.hist(arr1, bins=30, alpha=0.6, label="approachable (1)")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.legend()


st.title("iDetect - Investor Mood Detection")

col_a, col_b = st.columns([1, 2])

with col_a:
    st.subheader("Model status")
    try:
        r = requests.get(f"{DEFAULT_API_URL}/health", timeout=10)
        if r.status_code == 200:
            health = r.json()
            st.write(health)
            with st.expander("Retraining uploads (SQLite)"):
                try:
                    s = requests.get(f"{DEFAULT_API_URL}/retrain/uploads/summary", timeout=10)
                    if s.ok:
                        st.json(s.json())
                    else:
                        st.caption("Could not load upload summary.")
                except Exception:
                    st.caption("Could not load upload summary.")
        else:
            st.error(f"Health check failed: {r.status_code}")
    except Exception as e:
        st.error(f"Could not reach API: {e}")

    st.subheader("Upload for prediction")
    pred_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"], key="pred_upload")
    threshold = st.slider("Decision threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

    if pred_file is not None:
        if st.button("Predict"):
            img_bytes = pred_file.read()
            files = {"image": (pred_file.name, img_bytes, pred_file.type or "image/jpeg")}
            data = {"threshold": str(threshold)}
            resp = requests.post(f"{DEFAULT_API_URL}/predict", files=files, data=data, timeout=60)
            st.json(resp.json())

with col_b:
    st.subheader("Data visualizations & feature interpretations")
    counts = load_dataset_counts()
    fig1, ax1 = plt.subplots()
    ax1.bar(["not_approachable (0)", "approachable (1)"], [counts[0], counts[1]])
    ax1.set_title("Class distribution in dataset")
    ax1.set_ylabel("number of images")
    st.pyplot(fig1)

    st.markdown(
        """
### Interpretation (what story do these features tell?)
1. **Brightness**: if approachable faces are brighter (often linked with more “open” facial expression lighting), brightness shifts by class.
2. **Edge density**: more intense facial contours can produce higher edge density; expression style differences can show up here.
3. **Gradient magnitude**: captures overall “visual energy” (how strong pixel changes are). If one class has more sharp changes (e.g., tighter expressions), it will trend higher.
"""
    )

    with st.spinner("Computing feature histograms (sampled)..."):
        hists = compute_three_feature_hists(max_images_per_emotion=40, image_size=224)

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    plot_hist(axes[0], hists["edge_density"][0], hists["edge_density"][1], "Edge density", "fraction of edge pixels")
    plot_hist(axes[1], hists["brightness"][0], hists["brightness"][1], "Brightness", "mean grayscale intensity (0-255)")
    plot_hist(axes[2], hists["gradient_mag"][0], hists["gradient_mag"][1], "Gradient magnitude", "mean Sobel magnitude")
    st.pyplot(fig)

st.divider()
st.subheader("Bulk upload & retraining")

upload_label = st.selectbox("Label for uploaded images", options=[0, 1], format_func=lambda x: f"{x} ({'approachable' if x==1 else 'not_approachable'})")
upload_files = st.file_uploader(
    "Choose multiple images to add to the retraining dataset",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
    key="retrain_uploads",
)

if st.button("Upload selected data"):
    if not upload_files:
        st.warning("Choose one or more images first.")
    else:
        img_files = []
        for f in upload_files:
            img_files.append(
                ("images", (f.name, f.read(), f.type or "image/jpeg"))
            )
        data = {"label": str(upload_label)}
        resp = requests.post(f"{DEFAULT_API_URL}/upload", files=img_files, data=data, timeout=120)
        st.json(resp.json())
        if resp.ok:
            try:
                sum_resp = requests.get(f"{DEFAULT_API_URL}/retrain/uploads/summary", timeout=10)
                if sum_resp.ok:
                    st.caption("SQLite totals after upload:")
                    st.json(sum_resp.json())
            except Exception:
                pass

if st.button("Trigger retraining"):
    data = {
        "epochs": "5",
        "batch_size": "16",
        "train_base": "false",
        "learning_rate": "0.0001",
    }
    resp = requests.post(f"{DEFAULT_API_URL}/retrain", data=data, timeout=60)
    payload = resp.json()
    st.json(payload)
    job_id = payload.get("job_id")
    if job_id:
        status_placeholder = st.empty()
        poll_interval_s = 2.0
        max_wait_s = 3600.0
        deadline = time.monotonic() + max_wait_s
        while time.monotonic() < deadline:
            status_resp = requests.get(f"{DEFAULT_API_URL}/job/{job_id}", timeout=30)
            status_resp.raise_for_status()
            body = status_resp.json()
            status_placeholder.json(body)
            if body.get("status") in ("completed", "failed"):
                break
            time.sleep(poll_interval_s)
        else:
            st.warning(
                f"Stopped polling after {int(max_wait_s // 60)} minutes. "
                f"Job may still be running; open GET {DEFAULT_API_URL}/job/{job_id}"
            )


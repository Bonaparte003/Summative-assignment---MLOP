<p align="center">
  <img src="logo.png" styles="height:30px; width:auto" alt="iDetect logo" />
</p>

<h1 align="center">iDetect — MLOps pipeline</h1>

<p align="center"><strong>Investor mood / approachability from face images</strong></p>

---

## Links

| | |
|---|---|
| **Repository (this project)** | [github.com/Bonaparte003/Summative-assignment---MLOP](https://github.com/Bonaparte003/Summative-assignment---MLOP) |
| **Live app (hosted)** | [Streamlit UI — http://16.170.235.209](http://16.170.235.209)|
| **Video demo** | [YouTube walkthrough](https://www.youtube.com/watch?v=Qit_-G2jZ_w) |
| **Prior project (Introduction to ML summative)** | [github.com/Bonaparte003/iDetect-Summative-introduction_to_machine_learning](https://github.com/Bonaparte003/iDetect-Summative-introduction_to_machine_learning) |

The hosted stack matches **Docker Compose** locally: UI on **8501**  reverse-proxied to port **80**, API on **8000**. Open **8000** (and **8089** if you run Locust) in the instance **security group** if those URLs time out.

The earlier repository is the **introduction module** work (notebook on GitHub). **This** repository adds the pipeline: training notebook under `notebook/`, `src/`, FastAPI, Streamlit, SQLite upload log, Docker, and Locust.

---

## What this project does

**iDetect** predicts whether a face looks **approachable (1)** or **not approachable (0)** from an **image**, using folders mapped as:

- **Class 1:** `happy`, `neutral`
- **Class 0:** `anger`, `contempt`

You can **train offline** (notebook or `src/train.py`), **serve** predictions via **FastAPI**, use a **Streamlit** UI, **upload** extra images for **retraining**, and **load-test** `/predict` with **Locust**.

---

## Walkthrough (Docker Compose)

This is the recommended path: train once (notebook or CLI), place the weights on disk, then bring up **api**, **ui**, and optionally **locust** with Compose.

### Prerequisites

- **Docker** with Compose v2 (`docker compose`)
- **Git**
- **AffectNet** (or the same folder layout) on your machine when you train — [Kaggle — AffectNet](https://www.kaggle.com/datasets/mstjebashazida/affectnet)

### 1. Clone the repository

```bash
git clone https://github.com/Bonaparte003/Summative-assignment---MLOP
cd Summative-assignment---MLOP
```

### 2. Get your first model (pipeline notebook)

The API loads **`models/idetect_classifier.keras`** from the repo (mounted into the container). Produce that file once using the pipeline notebook:

1. **Environment** — Create a virtualenv (or use Jupyter/Colab) and install dependencies:  
   `python3 -m venv .venv && source .venv/bin/activate` then `pip install -r requirements.txt` (or install the same packages in Colab).

2. **Dataset layout** — Point training at a directory that contains **`happy/`**, **`neutral/`**, **`anger/`**, and **`contempt/`** (each folder holds images). Putting **AffectNet** at the **repo root** matches the defaults in the notebook.

   ```text
   AffectNet/
     happy/
     neutral/
     anger/
     contempt/
   ```

3. **Open** **`notebook/iDetect_Project_Summative_machine_learning_pipeline.ipynb`**.

4. **Set paths** in the configuration cells:
   - **`DATA_DIR`** — absolute path to the folder above (e.g. `.../Summative-assignment---MLOP/AffectNet`).
   - **`OUTPUT_DIR`** — where artifacts should be written (any writable folder, e.g. `.../iDetect_outputs`).

5. **Run all cells** through training and export. The notebook writes:
   - **`{OUTPUT_DIR}/models/idetect_classifier.keras`**
   - **`{OUTPUT_DIR}/reports/metrics.json`** (and related report files under `reports/`).

6. **Install the model into this repo for Docker** — Copy the saved file into the mounted model directory the stack expects:

   ```bash
   mkdir -p models reports
   cp /path/to/your/OUTPUT_DIR/models/idetect_classifier.keras models/
   cp /path/to/your/OUTPUT_DIR/reports/metrics.json reports/    # optional but useful for health/metadata
   ```

   **`src.train`** (CLI) also writes **`reports/model_meta.json`**; copy that into **`reports/`** if present so **`GET /health`** can show richer metadata. The pipeline notebook’s export section focuses on **`metrics.json`** and **`idetect_classifier.keras`**.

**CLI alternative (same artifacts):** from the repo root, with venv active:

```bash
python -m src.train --data_dir "AffectNet" --output_dir "."
```

That writes **`models/idetect_classifier.keras`**, **`reports/metrics.json`**, and **`reports/model_meta.json`** directly under the repo.

### 3. Run the stack

```bash
docker compose up --build
```

| Service | URL / port | Notes |
|--------|------------|--------|
| **API** | [http://localhost:8000/docs](http://localhost:8000/docs) | OpenAPI UI; health at **`GET /health`** |
| **UI** | [http://localhost:8501](http://localhost:8501) | Streamlit; uses **`API_URL=http://api:8000`** inside Compose |
| **Locust** | [http://localhost:8089](http://localhost:8089) | Preconfigured with **`--host http://api:8000`**; needs **`AffectNet/happy`** at **image build** time (see **`locust/Dockerfile`**) |

Compose **bind-mounts** **`./models`** and **`./data`** into the API container. Without **`models/idetect_classifier.keras`**, the API starts but **`/health`** reports **`model_missing`** and **`/predict`** returns **503** until the file is present.

**Retraining inside Docker** expects training data at **`/app/AffectNet`** in the API container. If `AffectNet` is not baked into the image (e.g. it is gitignored), add a volume when you need retrain, for example extend **`docker-compose.yml`** with  
`- ./AffectNet:/app/AffectNet` under **`api.volumes`**, or run a one-off **`docker run`** with that mount.

**Endpoints (summary):**

- Predict: **`POST /predict`** (multipart field **`image`**)
- Bulk upload (retraining): **`GET /retrain/bulk-upload-info`** documents **`POST /upload`**; the POST route is **omitted from interactive `/docs`**
- Retrain flow: **`POST /upload`** → **`POST /retrain`** → **`GET /job/{job_id}`**

### 4. Locust without the Compose service

If the API runs on the host (**`localhost:8000`**) instead of Compose:

```bash
export IMAGE_PATH=AffectNet/happy
locust -f locust/locustfile.py --host http://localhost:8000
```

Open **http://localhost:8089**, run a test, then export or screenshot stats for your write-up.

### 5. Alternative: run API + UI without Docker

**Terminal 1 — API**

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

**Terminal 2 — UI**

```bash
streamlit run ui/app.py --server.port 8501
```

Set **`API_URL`** if Streamlit is not on the same machine as the API (default **`http://localhost:8000`**). On **hosted** EC2 with Compose, the UI uses **`API_URL=http://api:8000`**; for a manual host-only setup use your public API base (e.g. **`http://16.170.235.209:8000`**).

### 6. API-only container (no Compose)

```bash
docker build -t idetect-api .
docker run -p 8000:8000 \
  -e DATA_DIR=/app/AffectNet \
  -e DATABASE_PATH=/app/data/idetect_retrain.sqlite3 \
  -v "$PWD/models:/app/models" \
  -v "$PWD/data:/app/data" \
  idetect-api
```

---

## Example metrics (`reports/metrics.json`)

After training, `reports/metrics.json` includes test-set scores. A recent local run reported approximately:

| Metric | Example value |
|--------|----------------|
| Accuracy | ~0.93 |
| Precision | ~0.99 |
| Recall | ~0.87 |
| F1 | ~0.92 |
| ROC-AUC | ~0.98 |

Exact numbers depend on your data and run; open **`reports/metrics.json`** for the current `metrics` block and `split_sizes`.

---

## Repository layout

| Path | Purpose |
|------|---------|
| `notebook/iDetect_Project_Summative_machine_learning_pipeline.ipynb` | Pipeline training, evaluation, export |
| `src/preprocessing.py`, `model.py`, `prediction.py` | Data, graph, inference |
| `src/train.py` | CLI training (used by API retrain) |
| `src/retrain_db.py` | SQLite metadata for upload API |
| `api/main.py` | FastAPI |
| `ui/app.py` | Streamlit |
| `locust/locustfile.py` | HTTP load test on `/predict` |
| `data/uploads/` | Retraining images (`0/` and `1/`) |
| `data/idetect_retrain.sqlite3` | Created at runtime (gitignored) — upload audit trail |

<p align="center"><sub>iDetect — MLOps summative</sub></p>

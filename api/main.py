import os
import threading
import uuid
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from src.prediction import Predictor
from src.retrain_db import init_db, insert_upload_row, list_recent_uploads, upload_summary
from src.train import train as train_fn


app = FastAPI(title="iDetect - Investor Mood Classifier API")

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR / "AffectNet")))
UPLOADS_DIR = Path(os.getenv("UPLOADS_DIR", str(BASE_DIR / "data" / "uploads")))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", str(BASE_DIR)))
DATABASE_PATH = Path(
    os.getenv("DATABASE_PATH", str((UPLOADS_DIR.parent / "idetect_retrain.sqlite3").resolve()))
)

MODEL_PATH = OUTPUT_DIR / "models" / "idetect_classifier.keras"
MODEL_META_PATH = OUTPUT_DIR / "reports" / "model_meta.json"
METRICS_PATH = OUTPUT_DIR / "reports" / "metrics.json"

_model_lock = threading.Lock()
_predictor: Optional[Predictor] = None

_job_lock = threading.Lock()
_jobs: Dict[str, Dict] = {}


def _load_predictor() -> None:
    global _predictor
    if not MODEL_PATH.exists():
        _predictor = None
        return
    _predictor = Predictor(model_path=str(MODEL_PATH))


@app.on_event("startup")
def _startup() -> None:
    # Ensure uploads directory exists so /upload works immediately.
    (UPLOADS_DIR / "0").mkdir(parents=True, exist_ok=True)
    (UPLOADS_DIR / "1").mkdir(parents=True, exist_ok=True)
    init_db(str(DATABASE_PATH))
    with _model_lock:
        _load_predictor()


@app.get("/health")
def health() -> JSONResponse:
    with _model_lock:
        model_ready = MODEL_PATH.exists() and _predictor is not None
        model_version = MODEL_META_PATH.exists() and MODEL_META_PATH.read_text(encoding="utf-8")
    db_summary = upload_summary(str(DATABASE_PATH))
    return JSONResponse(
        {
            "status": "ok" if model_ready else "model_missing",
            "model_path": str(MODEL_PATH),
            "model_ready": bool(model_ready),
            "model_meta": model_version if model_ready else None,
            "retrain_uploads_db": str(DATABASE_PATH),
            "retrain_uploads_total": db_summary["total"],
        }
    )


def _require_predictor() -> Predictor:
    with _model_lock:
        if _predictor is None:
            raise HTTPException(status_code=503, detail="Model not loaded. Train/retrain first.")
        return _predictor


@app.post("/predict")
def predict(
    image: UploadFile = File(...),
    threshold: float = Form(0.5),
) -> dict:
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image file.")
    content = image.file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file upload.")

    predictor = _require_predictor()
    with _model_lock:
        return predictor.predict(image_bytes=content, threshold=threshold)


@app.post("/upload")
def upload_images(
    label: int = Form(..., description="Binary label: 1=approachable, 0=not_approachable"),
    images: list[UploadFile] = File(...),
) -> dict:
    if label not in (0, 1):
        raise HTTPException(status_code=400, detail="label must be 0 or 1")
    if not images:
        raise HTTPException(status_code=400, detail="No files provided")

    target_dir = UPLOADS_DIR / str(label)
    target_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    db_ids: list[int] = []
    for f in images:
        content = f.file.read()
        if not content:
            continue
        fname = f"{uuid.uuid4().hex}_{Path(f.filename).name if f.filename else 'upload'}.jpg"
        out_path = target_dir / fname
        out_path.write_bytes(content)
        row_id = insert_upload_row(
            str(DATABASE_PATH),
            label,
            str(out_path.resolve()),
            f.filename,
        )
        db_ids.append(row_id)
        saved += 1

    if saved == 0:
        raise HTTPException(status_code=400, detail="No images were saved. Check your uploads.")

    return {
        "saved": saved,
        "label": label,
        "uploads_dir": str(target_dir),
        "sqlite_path": str(DATABASE_PATH),
        "sqlite_row_ids": db_ids,
        "sqlite_summary": upload_summary(str(DATABASE_PATH)),
    }


def _retrain_job(
    job_id: str,
    epochs: int,
    batch_size: int,
    train_base: bool,
    unfreeze_last_n: int,
    learning_rate: float,
) -> None:
    """
    Background retraining job.
    """
    try:
        uploads_present = UPLOADS_DIR.exists() and any(
            (UPLOADS_DIR / str(lbl)).exists() and any((UPLOADS_DIR / str(lbl)).iterdir())
            for lbl in (0, 1)
        )
        uploads_dir = str(UPLOADS_DIR) if uploads_present else None

        payload = train_fn(
            data_dir=str(DATA_DIR),
            uploads_dir=uploads_dir,
            output_dir=str(OUTPUT_DIR),
            epochs=epochs,
            batch_size=batch_size,
            image_size=224,
            limit_per_emotion=400,
            train_base=train_base,
            unfreeze_last_n=unfreeze_last_n,
            learning_rate=learning_rate,
            seed=42,
        )

        with _model_lock:
            _load_predictor()

        with _job_lock:
            _jobs[job_id]["status"] = "completed"
            _jobs[job_id]["result"] = {
                "run_id": payload.get("run_id"),
                "test_metrics": payload.get("metrics"),
            }
    except Exception as e:
        with _job_lock:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = str(e)


@app.get("/retrain/uploads/summary")
def retrain_uploads_summary() -> dict:
    return upload_summary(str(DATABASE_PATH))


@app.get("/retrain/uploads/recent")
def retrain_uploads_recent(limit: int = 50) -> dict:
    return {"items": list_recent_uploads(str(DATABASE_PATH), limit=limit)}


@app.post("/retrain")
def retrain(
    epochs: int = Form(5),
    batch_size: int = Form(16),
    train_base: bool = Form(False),
    unfreeze_last_n: int = Form(30),
    learning_rate: float = Form(1e-4),
) -> dict:
    job_id = uuid.uuid4().hex
    with _job_lock:
        _jobs[job_id] = {"status": "running"}

    n_unfreeze = 0 if train_base else max(0, unfreeze_last_n)

    t = threading.Thread(
        target=_retrain_job,
        args=(job_id, epochs, batch_size, train_base, n_unfreeze, learning_rate),
        daemon=True,
    )
    t.start()
    return {"job_id": job_id, "status": "running"}


@app.get("/job/{job_id}")
def job(job_id: str) -> dict:
    with _job_lock:
        if job_id not in _jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        return _jobs[job_id]


"""FastAPI app: three endpoints over the Celery job queue in worker.py.
No auth, no database -- job state and results live in Redis via Celery's
own result backend (see worker.py).

Run directly (not via Docker):
    uvicorn api.main:app --host 0.0.0.0 --port 8000

Run in docker-compose: see the `api` service there.
"""

from __future__ import annotations

from typing import Optional

from celery.result import AsyncResult
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .worker import celery_app, process_book

app = FastAPI(title="tibetan-layer-detection-pipeline API")

# Celery's own states, mapped to the four requested here.
_STATUS_MAP = {
    "PENDING": "queued",
    "STARTED": "running",
    "PROGRESS": "running",
    "SUCCESS": "complete",
    "FAILURE": "failed",
    "RETRY": "running",
}


class ProcessRequest(BaseModel):
    book_path: str
    # Optional[...], not list[str] | None: Pydantic evaluates this
    # annotation at runtime to build its validator, and the `|` union
    # syntax isn't valid on Python 3.9 (this package's declared minimum).
    layers: Optional[list[str]] = None


@app.post("/process", status_code=202)
def process(req: ProcessRequest) -> dict:
    """Queues one book for detection. book_path is resolved relative to
    TIBETAN_DATA_DIR (see config.py / the worker container's /data mount)."""
    task = process_book.delay(req.book_path, req.layers)
    return {"job_id": task.id}


@app.get("/status/{job_id}")
def status(job_id: str) -> dict:
    """status: queued | running | complete | failed. stage and book_id are
    only populated once the worker has started (state PROGRESS or later)."""
    r = AsyncResult(job_id, app=celery_app)
    out = {"job_id": job_id, "status": _STATUS_MAP.get(r.state, r.state.lower()), "stage": None, "book_id": None}
    if r.state == "PROGRESS" and isinstance(r.info, dict):
        out["stage"] = r.info.get("stage")
        out["book_id"] = r.info.get("book_id")
    elif r.state == "SUCCESS" and isinstance(r.result, dict):
        out["stage"] = "complete"
        out["book_id"] = r.result.get("book_id")
    elif r.state == "FAILURE":
        out["stage"] = "failed"
    return out


@app.get("/results/{job_id}")
def results(job_id: str) -> dict:
    """The detect()-shaped result once the job is complete. 409 while it's
    still queued or running; the underlying error if it failed."""
    r = AsyncResult(job_id, app=celery_app)
    if r.state == "SUCCESS":
        return r.result
    if r.state == "FAILURE":
        raise HTTPException(status_code=500, detail=f"job failed: {r.info}")
    raise HTTPException(status_code=409, detail=f"job not finished (status: "
                                                f"{_STATUS_MAP.get(r.state, r.state.lower())})")

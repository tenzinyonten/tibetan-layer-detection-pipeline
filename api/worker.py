"""Celery worker: one task, process_book, that calls detect() from
tibetan_layer_detection for a single book and reports per-stage progress
via detect()'s on_stage callback.

Redis is both the Celery broker and the result backend, so job status and
the final result live entirely in Redis -- no separate database, matching
the "Redis stores everything" brief.

Run directly (not via Docker):
    celery -A api.worker worker --loglevel=info

Run in docker-compose: see the `worker` service there.
"""

from __future__ import annotations

from celery import Celery

from tibetan_layer_detection import detect

from .config import DATA_DIR, DEVICE, REDIS_URL

celery_app = Celery("tibetan_layer_detection_api", broker=REDIS_URL, backend=REDIS_URL)
celery_app.conf.update(
    task_track_started=True,
    result_expires=None,  # "Redis stores everything": don't auto-expire results
)


@celery_app.task(bind=True, name="api.worker.process_book")
def process_book(self, book_path: str, layers: list | None = None) -> dict:
    """book_path is resolved relative to DATA_DIR (see config.py). layers
    is a list of layer names, or None/omitted for every configured layer
    (detect()'s own default)."""
    full_path = DATA_DIR / book_path
    if not full_path.exists():
        raise FileNotFoundError(f"{full_path} does not exist (DATA_DIR={DATA_DIR}, "
                                f"book_path={book_path!r})")

    def on_stage(stage: str, meta: dict) -> None:
        book_ids = meta.get("book_ids") or []
        self.update_state(state="PROGRESS", meta={
            "stage": stage,
            "book_id": book_ids[0] if book_ids else None,
        })

    result = detect(str(full_path), layers=layers or "all", device=DEVICE, on_stage=on_stage)
    # detect() returns a single dict for a file input (which book_path
    # always is here, per the API's one-book-per-job design), or a list for
    # a directory. process_book only ever submits a single book_path, but
    # guard anyway rather than assume.
    return result if isinstance(result, dict) else (result[0] if result else {})

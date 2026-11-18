"""Stage 3 JSON output."""

from __future__ import annotations

import json
from pathlib import Path

from .review import needs_review


def _span_payload(s: dict, name: str, review_threshold: float, ta_threshold: float | None = None) -> dict:
    out = {"start": s["start"], "end": s["end"],
           "confidence": None if s["confidence"] is None else round(s["confidence"], 4)}
    for key in ("block", "model"):  # title/author extras
        if key in s:
            out[key] = s[key]
    out["review_needed"] = needs_review(s, name, review_threshold, ta_threshold)
    return out


def write_json(book_id: str, text_length: int, layers: dict[str, list[dict]],
              errors: dict[str, str], out_dir: Path, review_threshold: float,
              ta_threshold: float | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{book_id}.json"
    payload = {
        "book_id": book_id,
        "text_length": text_length,
        "layers": {name: [_span_payload(s, name, review_threshold, ta_threshold) for s in spans]
                  for name, spans in layers.items()},
    }
    if errors:
        payload["errors"] = errors
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path

"""Stage 3 review flags: which review threshold applies to a layer, and whether a span needs review."""

from __future__ import annotations

from ..layer_config import is_title_author


def threshold_for(layer_name: str, review_threshold: float, ta_threshold: float | None = None) -> float:
    """review_threshold is the --review-threshold value, one number for every
    layer. For the title/author layers, ta_threshold (--title-author-review-threshold)
    wins over it."""
    if ta_threshold is not None and is_title_author(layer_name):
        return ta_threshold
    return review_threshold


def needs_review(s: dict, name: str, review_threshold: float, ta_threshold: float | None = None) -> bool:
    """A span is flagged when its confidence is below the threshold. A span with
    no confidence is flagged only if it carries review_needed itself."""
    if s.get("confidence") is None:
        return bool(s.get("review_needed"))
    return s["confidence"] < threshold_for(name, review_threshold, ta_threshold)

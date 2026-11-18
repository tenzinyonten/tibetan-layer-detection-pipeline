"""Stage 3 span filters: short spans, non-Tibetan spans, low confidence, tsawa inside quotations."""

from __future__ import annotations

from ..decode import has_tibetan
from ..layer_config import LAYERS, is_title_author


# Spans under this many characters are dropped from every output (JSON, HTML,
# .opf) for all layers. On the Sabche test set the 19 predicted spans of 1-4
# characters matched no gold span, and removing them moved F1 from 0.962 to
# 0.964. Same threshold as decode.MIN_SPAN_CHARS (inference-side filter);
# applied again here so predictions made before that filter existed, or by
# another tool, are cleaned too. Spans are end-exclusive, so length = end - start.
MIN_SPAN_CHARS = 5


def drop_short_spans(layers: dict[str, list[dict]]) -> tuple[dict[str, list[dict]], dict[str, int]]:
    """Returns (layers without spans under MIN_SPAN_CHARS, n_dropped per layer).
    Title and author are exempt: they are one span per work and a short
    author name is real, not a fragment."""
    kept = {n: spans if is_title_author(n) else [s for s in spans if s["end"] - s["start"] >= MIN_SPAN_CHARS]
            for n, spans in layers.items()}
    return kept, {n: len(layers[n]) - len(kept[n]) for n in layers}


def drop_non_tibetan_spans(layers: dict[str, list[dict]], text: str) -> tuple[dict[str, list[dict]], dict[str, int]]:
    """Returns (layers without spans that contain no Tibetan character, n_dropped per layer).
    Catches scraping artifacts (e.g. English navigation text) in every layer."""
    kept = {n: [s for s in spans if has_tibetan(text[s["start"]:s["end"]])] for n, spans in layers.items()}
    return kept, {n: len(layers[n]) - len(kept[n]) for n in layers}


def drop_low_confidence(layers: dict[str, list[dict]]) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    """For each layer whose LayerConfig.min_confidence is set (Sabche: 0.85), drop the spans whose
    confidence is below it. Returns (layers, dropped spans per layer, each kept as it was). A span with no
    confidence is never dropped, and layers without the option are untouched."""
    out, dropped = {}, {}
    for name, spans in layers.items():
        cut = LAYERS[name].min_confidence if name in LAYERS else None
        if cut is None:
            out[name] = spans
            continue
        out[name] = [s for s in spans if s.get("confidence") is None or s["confidence"] >= cut]
        gone = [s for s in spans if s.get("confidence") is not None and s["confidence"] < cut]
        if gone:
            dropped[name] = gone
    return out, dropped


# Drop a predicted tsawa span when this much of it sits inside a *predicted*
# quotation span (Bo-Quotation-Detection). Never reads gold Quotation.yml /
# Citation.yml -- incoming books are unannotated. If the quotation model
# produced no spans, this is a no-op. Tested on tsawa only (v10 model, test
# benchmark: F1 0.842 -> 0.890); not validated for any other layer.
QUOTE_COVERAGE_DROP = 0.5


def _union(spans: list[dict]) -> list[tuple[int, int]]:
    """Merge overlapping or touching spans into disjoint (start, end) intervals,
    so overlapping quotation spans are not double counted."""
    merged: list[list[int]] = []
    for s, e in sorted((sp["start"], sp["end"]) for sp in spans):
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def _coverage(span: dict, others: list[dict]) -> float:
    """Fraction of `span`'s characters inside the union of `others`."""
    s, e = span["start"], span["end"]
    if e <= s:
        return 0.0
    return sum(max(0, min(e, oe) - max(s, os_)) for os_, oe in _union(others)) / (e - s)


def split_quoted_tsawa(
    layers: dict[str, list[dict]], min_coverage: float = QUOTE_COVERAGE_DROP,
) -> tuple[dict[str, list[dict]], list[dict]]:
    """Drop tsawa spans whose character coverage by the union of the predicted
    quotation spans is at least `min_coverage`. Uses `layers["quotation"]` from
    Stage 2 only, never gold. Only tsawa spans are dropped; quotation is left
    alone. No-op (the same `layers` object comes back) if either layer is
    missing or empty, or nothing reaches the threshold. Returns (layers,
    dropped) where each dropped span keeps its fields plus "quote_coverage"."""
    tsawa = layers.get("tsawa") or []
    quotes = layers.get("quotation") or []
    if not tsawa or not quotes:
        return layers, []
    kept, dropped = [], []
    for s in tsawa:
        cov = _coverage(s, quotes)
        if cov >= min_coverage:
            dropped.append({**s, "quote_coverage": round(cov, 4)})
        else:
            kept.append(s)
    if not dropped:
        return layers, []
    out = dict(layers)
    out["tsawa"] = kept
    return out, dropped


def drop_quoted_tsawa(
    layers: dict[str, list[dict]], min_coverage: float = QUOTE_COVERAGE_DROP,
) -> tuple[dict[str, list[dict]], int]:
    """split_quoted_tsawa, returning only how many tsawa spans were dropped."""
    out, dropped = split_quoted_tsawa(layers, min_coverage)
    return out, len(dropped)

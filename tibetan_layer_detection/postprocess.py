#!/usr/bin/env python3
"""postprocess.py -- Stage 3 of the layer-detection pipeline: rendering.

Joins Stage 2's predictions/<book_id>.json (spans + confidence) with Stage
1's preprocessed/<book_id>.json (source text + source_path -- predictions
alone carry neither) and writes final JSON, HTML, and/or OpenPecha .opf
output, plus a batch summary and a run log.

--opf writes predictions as OpenPecha-shaped layer YAML, only for layers with
at least one span. A plain .txt input gets a fresh minimal structure under
--out: <id>.opf/<id>.opf/base/v001.txt plus layers/v001/. A real OpenPecha
book (Stage 1 source_path matches <id>.opf/<id>.opf/base/v001.txt) whose
layers/v001/ already exists holds gold annotations, so its predictions go to
the sibling layers/predicted/ instead and layers/v001/ is never touched.
Schema verified against real files in this repo
before writing any code: data/raw_opf/P000201.opf's Tsawa.yml/Sabche.yml/
Chapter.yml, data/raw_opf/P000172.opf's Quotation.yml, data/raw_opf/
I058DD999.opf's Yigchung.yml. Deliberately does not add a confidence field
there, to keep an exact structural match with those files; confidence lives
in the JSON output only.

pyyaml is a required dependency of this package (see pyproject.toml), so,
unlike an earlier standalone version of this tool where PyYAML was made an
optional lazy import, it is imported normally here at module level.

Usage (as an installed console script)
---------------------------------------
    tibetan-postprocess --input predictions/ --source preprocessed/ \\
        --out output/ --json --html --opf --review-threshold 0.7
"""

from __future__ import annotations

import argparse
import html as html_mod
import json
import re
import sys
import uuid
from bisect import bisect_left
from datetime import datetime
from pathlib import Path

import yaml

from .decode import has_tibetan
from .layer_config import LAYERS, base_layer, is_title_author

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


# ---------------------------------------------------------------------------
# joining Stage 1 + Stage 2
# ---------------------------------------------------------------------------

def load_pair(book_id: str, predictions_dir: Path, preprocessed_dir: Path) -> tuple[dict, dict | None]:
    """Returns (predictions, preprocessed_or_None). preprocessed is None if
    Stage 1's file for this book_id is missing -- the caller should warn and
    skip JSON/HTML/opf output for that book, since none of them can be
    produced without at least the source text/length."""
    predictions = json.loads((predictions_dir / f"{book_id}.json").read_text(encoding="utf-8"))
    src_path = preprocessed_dir / f"{book_id}.json"
    preprocessed = json.loads(src_path.read_text(encoding="utf-8")) if src_path.is_file() else None
    return predictions, preprocessed


# ---------------------------------------------------------------------------
# review flags
# ---------------------------------------------------------------------------

def threshold_for(layer_name: str, review_threshold: float, ta_threshold: float | None = None) -> float:
    """review_threshold is the --review-threshold value, one number for every
    layer. For the title/author layers, ta_threshold (--title-author-review-threshold)
    wins over it."""
    if ta_threshold is not None and is_title_author(layer_name):
        return ta_threshold
    return review_threshold


def needs_review(s: dict, name: str, review_threshold: float, ta_threshold: float | None = None) -> bool:
    """A span is flagged when its confidence is below the threshold. A skill span
    has no confidence: the skill's own checks (review_needed from Stage 2) decide."""
    if s.get("confidence") is None:
        return bool(s.get("review_needed"))
    return s["confidence"] < threshold_for(name, review_threshold, ta_threshold)


def _span_payload(s: dict, name: str, review_threshold: float, ta_threshold: float | None = None) -> dict:
    out = {"start": s["start"], "end": s["end"],
           "confidence": None if s["confidence"] is None else round(s["confidence"], 4)}
    for key in ("block", "system", "model", "review_reasons", "systems_agree"):  # title/author extras
        if key in s:
            out[key] = s[key]
    out["review_needed"] = needs_review(s, name, review_threshold, ta_threshold)
    return out


# ---------------------------------------------------------------------------
# output: JSON
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# output: OpenPecha layer YAML (--opf)
# ---------------------------------------------------------------------------

def opf_root_for(source_path: Path) -> Path | None:
    """<id>.opf/<id>.opf/base/v001.txt -> the inner <id>.opf directory (the
    real OpenPecha root, where layers/v001/ lives), or None if source_path
    doesn't match that exact layout."""
    if source_path.name == "v001.txt" and source_path.parent.name == "base":
        grandparent = source_path.parent.parent
        if grandparent.name.endswith(".opf"):
            return grandparent
    return None


def normalize_tsegs(text: str) -> str:
    text = re.sub(r'\s+་', '་ ', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()


def _sub_tracked(pattern: str, new: str, text: str, src: list[int], pick) -> tuple[str, list[int]]:
    out, out_src, pos = [], [], 0
    for m in re.finditer(pattern, text):
        out.append(text[pos:m.start()])
        out_src.extend(src[pos:m.start()])
        out.append(new)
        out_src.extend(pick(m, src))
        pos = m.end()
    out.append(text[pos:])
    out_src.extend(src[pos:])
    return "".join(out), out_src


def normalize_tsegs_tracked(text: str) -> tuple[str, list[int]]:
    """Same result as normalize_tsegs, plus for every output character the
    offset of the input character it came from (non-decreasing)."""
    src = list(range(len(text)))
    text, src = _sub_tracked(r'\s+་', '་ ', text, src,
                             lambda m, s: [s[m.start()], s[m.end() - 1]])
    text, src = _sub_tracked(r' +', ' ', text, src, lambda m, s: [s[m.start()]])
    lead = len(text) - len(text.lstrip())
    text = text.strip()
    return text, src[lead:lead + len(text)]


def remap_layers(layers: dict[str, list[dict]], src: list[int]) -> dict[str, list[dict]]:
    """Move span offsets from the original text to the normalized one. A span
    that only covered removed whitespace disappears."""
    out = {}
    for name, spans in layers.items():
        kept = []
        for s in spans:
            start, end = bisect_left(src, s["start"]), bisect_left(src, s["end"])
            if end > start:
                kept.append({**s, "start": start, "end": end})
        out[name] = kept
    return out


def write_plain_opf(book_id: str, text: str, out_dir: Path) -> Path:
    """For a plain .txt input: create <out>/<id>.opf/<id>.opf/ with
    base/v001.txt (the text given) and return that inner root, ready for
    write_opf_layers (which fills layers/v001/ via write_opf_layers)."""
    root = out_dir / f"{book_id}.opf" / f"{book_id}.opf"
    (root / "base").mkdir(parents=True, exist_ok=True)
    (root / "base" / "v001.txt").write_text(text, encoding="utf-8")
    return root


def write_opf_layers(layers: dict[str, list[dict]], opf_root: Path,
                     subdir: str = "v001") -> list[Path]:
    """Write predictions as OpenPecha-shaped layer YAML under
    layers/<subdir>/, skipping layers with no spans (see module docstring for
    which subdir is used when, and the real files this schema was verified
    against)."""
    out_dir = opf_root / "layers" / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, spans in layers.items():
        if not spans:
            continue
        layer_name = name.capitalize()
        data = {
            "id": uuid.uuid4().hex,
            "annotation_type": layer_name,
            "revision": "00001",
            "annotations": {
                uuid.uuid4().hex: {"span": {"start": int(s["start"]), "end": int(s["end"])}}
                for s in spans
            },
        }
        p = out_dir / f"{layer_name}.yml"
        p.write_text(yaml.safe_dump(data, default_flow_style=False, sort_keys=False,
                                    allow_unicode=True), encoding="utf-8")
        written.append(p)
    return written


# ---------------------------------------------------------------------------
# output: HTML (stacked underlines for overlapping layers)
# ---------------------------------------------------------------------------

def write_html(book_id: str, text: str, layers: dict[str, list[dict]],
               errors: dict[str, str], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{book_id}.html"

    events = []  # (char_pos, delta, layer_name) delta +1 open, -1 close
    for name, spans in layers.items():
        for s in spans:
            events.append((s["start"], 1, name))
            events.append((s["end"], -1, name))
    bounds = sorted({p for p, _, _ in events} | {0, len(text)})

    active: set[str] = set()
    open_at = {p: [] for p in bounds}
    close_at = {p: [] for p in bounds}
    for p, d, name in events:
        (open_at if d == 1 else close_at)[p].append(name)

    parts = []
    for i in range(len(bounds) - 1):
        p, nxt = bounds[i], bounds[i + 1]
        for name in close_at.get(p, ()):
            active.discard(name)
        for name in open_at.get(p, ()):
            active.add(name)
        chunk = html_mod.escape(text[p:nxt])
        if not chunk:
            continue
        if active:
            shadows = ", ".join(f"inset 0 -{2 + 2 * i}px {LAYERS[base_layer(n)].color}"
                                for i, n in enumerate(sorted(active)))
            title = ", ".join(sorted(active))
            parts.append(f'<span class="span" style="box-shadow:{shadows}" title="{title}">{chunk}</span>')
        else:
            parts.append(chunk)
    body = "".join(parts)

    counts = {name: len(spans) for name, spans in layers.items()}
    legend = "".join(
        f'<div class="legend-row"><span class="swatch" style="background:{LAYERS[base_layer(n)].color}"></span>'
        f'{n} <span class="mut">({counts.get(n, 0)} spans'
        + (f", {errors[n]}" if n in errors else "")
        + ')</span></div>'
        for n in layers.keys() | errors.keys()
    )

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>{html_mod.escape(book_id)} — layer detection</title>
<style>
body {{ font: 15px/1.9 system-ui, sans-serif; max-width: 900px; margin: 24px auto; padding: 0 16px; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 16px; margin-bottom: 16px; padding: 8px 12px;
          border: 1px solid #ddd; border-radius: 6px; }}
.legend-row {{ display: flex; align-items: center; gap: 6px; }}
.swatch {{ width: 14px; height: 14px; border-radius: 3px; display: inline-block; }}
.mut {{ color: #777; font-size: 13px; }}
.text {{ white-space: pre-wrap; word-break: break-word;
        font-family: "Noto Serif Tibetan", "Jomolhari", serif; font-size: 17px; }}
.span {{ padding: 0 1px; }}
</style></head><body>
<h1>{html_mod.escape(book_id)}</h1>
<div class="legend">{legend}</div>
<div class="text">{body}</div>
</body></html>"""
    path.write_text(page, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", type=Path, required=True, help="folder of Stage 2 predictions JSONs")
    ap.add_argument("--source", type=Path, required=True, help="folder of Stage 1 preprocessed JSONs")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--html", action="store_true")
    ap.add_argument("--opf", action="store_true",
                    help="also write predictions as OpenPecha layer YAML under "
                         "layers/v001/ for plain .txt input, or layers/predicted/ for a real "
                         "OpenPecha book whose layers/v001/ already exists (only layers with spans); only for books "
                         "whose Stage 1 source_path matches <id>.opf/<id>.opf/base/v001.txt")
    ap.add_argument("--keep-quoted-tsawa", action="store_true",
                    help="do not drop tsawa spans that mostly overlap a predicted "
                         "quotation span. Default: drop when coverage >= 0.5 and "
                         "the quotation *model* produced spans. Does not use gold.")
    ap.add_argument("--title-author-review-threshold", type=float, default=None,
                    help="a title/author span with confidence below this is flagged review_needed "
                         "(wins over --review-threshold for those layers). Default: --review-threshold.")
    ap.add_argument("--title-author-opf", action="store_true",
                    help="NOT IMPLEMENTED: writing title/author as OpenPecha layers (the layer names are not "
                         "known). Switching it on is an error; --opf alone skips these layers.")
    ap.add_argument("--review-threshold", type=float, default=0.7,
                    help="a span with confidence below this is flagged review_needed")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.title_author_opf:
        raise SystemExit("--title-author-opf is not implemented: the OpenPecha layer names for title and "
                         "author are not known. Leave it off; --opf writes the other layers only.")
    files = sorted(args.input.glob("*.json"))
    if not files:
        raise SystemExit(f"no predictions JSON files found in {args.input}")

    summary = []
    for i, fp in enumerate(files, 1):
        predictions = json.loads(fp.read_text(encoding="utf-8"))
        book_id = predictions["book_id"]
        _, preprocessed = load_pair(book_id, args.input, args.source)
        if preprocessed is None:
            print(f"  [warn] {book_id}: no matching Stage 1 file in {args.source}; "
                  f"skipping (need source text/length)", file=sys.stderr)
            continue

        layers, errors = predictions["layers"], predictions.get("errors", {})
        dropped_quoted = []
        if not args.keep_quoted_tsawa:
            layers, dropped_quoted = split_quoted_tsawa(layers)
            if dropped_quoted:
                print(f"  [note] {book_id}: dropped {len(dropped_quoted)} tsawa span(s) "
                      f"with >= {QUOTE_COVERAGE_DROP:.0%} predicted-quotation coverage",
                      file=sys.stderr)
        layers, dropped_short = drop_short_spans(layers)
        if any(dropped_short.values()):
            print(f"  [note] {book_id}: dropped spans under {MIN_SPAN_CHARS} chars: "
                  f"{ {n: c for n, c in dropped_short.items() if c} }", file=sys.stderr)
        text, text_length = preprocessed["text"], preprocessed["char_count"]
        layers, dropped_non_tibetan = drop_non_tibetan_spans(layers, text)
        if any(dropped_non_tibetan.values()):
            print(f"  [note] {book_id}: dropped spans with no Tibetan text: "
                  f"{ {n: c for n, c in dropped_non_tibetan.items() if c} }", file=sys.stderr)

        if args.json:
            p = write_json(book_id, text_length, layers, errors, args.out, args.review_threshold,
                           args.title_author_review_threshold)
            print(f"[{i}/{len(files)}] wrote {p}")
        if args.html:
            p = write_html(book_id, text, layers, errors, args.out)
            print(f"[{i}/{len(files)}] wrote {p}")
        if args.opf:
            source_path = Path(preprocessed["source_path"])
            root = opf_root_for(source_path)
            plain = root is None
            opf_layers = {n: sp for n, sp in layers.items() if not is_title_author(n)}
            if any(is_title_author(n) and sp for n, sp in layers.items()):
                print(f"  [note] {book_id}: title/author are not written to .opf (not implemented)", file=sys.stderr)
            if plain:
                if source_path.is_file():
                    norm_text, src = normalize_tsegs_tracked(text)
                    root = write_plain_opf(book_id, norm_text, args.out)
                    opf_layers = remap_layers(opf_layers, src)
                else:
                    print(f"  [warn] {book_id}: {source_path} not found; "
                          f"skipping --opf output for this book", file=sys.stderr)
            if root is not None:
                subdir = "predicted" if not plain and (root / "layers" / "v001").is_dir() else "v001"
                for p in write_opf_layers(opf_layers, root, subdir):
                    print(f"[{i}/{len(files)}] wrote {p}")

        review_needed = {n: sum(1 for s in spans
                               if needs_review(s, n, args.review_threshold, args.title_author_review_threshold))
                         for n, spans in layers.items()}
        combined_errors = dict(errors)
        if preprocessed.get("errors"):
            combined_errors["preprocess"] = preprocessed["errors"]
        summary.append({
            "book_id": book_id, "text_length": text_length,
            "n_windows": predictions.get("n_windows", {}), "elapsed_s": predictions.get("elapsed_s", {}),
            "spans": {n: len(spans) for n, spans in layers.items()},
            "review_needed": review_needed,
            "short_spans_dropped_at_render": dropped_short,
            "non_tibetan_spans_dropped_at_render": dropped_non_tibetan,
            "quoted_tsawa_dropped": len(dropped_quoted),
            "quoted_tsawa_dropped_spans": dropped_quoted,
            "errors": combined_errors,
        })

    print(f"\n{'='*60}\nBATCH SUMMARY — {len(summary)} documents")
    for row in summary:
        errs = f"  errors: {row['errors']}" if row["errors"] else ""
        print(f"  {row['book_id']:20} {row['text_length']:8,} chars  "
              f"{row['spans']}  review_needed={row['review_needed']}{errs}")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "_batch_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.out / '_batch_summary.json'}")

    run_log = args.out / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    run_log.write_text(json.dumps({
        "review_threshold": args.review_threshold,
        "title_author_review_threshold": args.title_author_review_threshold,
        "books": summary,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {run_log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

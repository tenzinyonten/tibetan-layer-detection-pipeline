#!/usr/bin/env python3
"""postprocess.py -- Stage 3 of the layer-detection pipeline: rendering.

Joins Stage 2's predictions/<book_id>.json (spans + confidence) with Stage
1's preprocessed/<book_id>.json (source text + source_path -- predictions
alone carry neither) and writes final JSON, HTML, and/or OpenPecha .opf
output, plus a batch summary and a run log. No model is loaded here; this
stage is pure formatting and can be re-run cheaply, including with a
different --review-threshold, without re-running inference.

--opf writes predictions as OpenPecha-shaped layer YAML under
layers/predicted/, a SIBLING of the real layers/v001/, never that folder
itself -- layers/v001/ holds the gold annotations the source project's whole
dataset pipeline reads as ground truth, and this tool must not be able to
silently overwrite that. Only fires for a book whose Stage 1 source_path
matches the real OpenPecha layout (<id>.opf/<id>.opf/base/v001.txt); anything
else is skipped with a warning. Schema verified against real files in
tenzinyonten/layer_detection before writing any code: data/raw_opf/
P000201.opf's Tsawa.yml/Sabche.yml/Chapter.yml, data/raw_opf/P000172.opf's
Quotation.yml, data/raw_opf/I058DD999.opf's Yigchung.yml. Deliberately does
not add a confidence field there, to keep an exact structural match with
those files; confidence lives in the JSON output only.

Usage
-----
    python postprocess.py --input predictions/ --source preprocessed/ \\
        --out output/ --json --html --opf --review-threshold 0.7
"""

from __future__ import annotations

import argparse
import html as html_mod
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from layer_config import LAYERS  # noqa: E402


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
# output: JSON
# ---------------------------------------------------------------------------

def write_json(book_id: str, text_length: int, layers: dict[str, list[dict]],
              errors: dict[str, str], out_dir: Path, review_threshold: float) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{book_id}.json"
    payload = {
        "book_id": book_id,
        "text_length": text_length,
        "layers": {name: [{"start": s["start"], "end": s["end"], "confidence": round(s["confidence"], 4),
                          "review_needed": s["confidence"] < review_threshold}
                         for s in spans]
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


def write_opf_layers(layers: dict[str, list[dict]], opf_root: Path) -> list[Path]:
    """Write predictions as OpenPecha-shaped layer YAML under
    layers/predicted/ (see module docstring for the safety rationale and the
    real files this schema was verified against). Imports PyYAML lazily,
    here only, so `pip install transformers torch` (this repo's declared,
    tested requirement) stays sufficient for ordinary --json/--html use;
    PyYAML is only needed when --opf is actually passed."""
    try:
        import yaml
    except ImportError:
        raise SystemExit("--opf needs PyYAML: pip install pyyaml")
    out_dir = opf_root / "layers" / "predicted"
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
            shadows = ", ".join(f"inset 0 -{2 + 2 * i}px {LAYERS[n].color}"
                                for i, n in enumerate(sorted(active)))
            title = ", ".join(sorted(active))
            parts.append(f'<span class="span" style="box-shadow:{shadows}" title="{title}">{chunk}</span>')
        else:
            parts.append(chunk)
    body = "".join(parts)

    counts = {name: len(spans) for name, spans in layers.items()}
    legend = "".join(
        f'<div class="legend-row"><span class="swatch" style="background:{LAYERS[n].color}"></span>'
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
                         "layers/predicted/ (never layers/v001/); only for books "
                         "whose Stage 1 source_path matches <id>.opf/<id>.opf/base/v001.txt")
    ap.add_argument("--review-threshold", type=float, default=0.7,
                    help="a span with confidence below this is flagged review_needed")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
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
        text, text_length = preprocessed["text"], preprocessed["char_count"]

        if args.json:
            p = write_json(book_id, text_length, layers, errors, args.out, args.review_threshold)
            print(f"[{i}/{len(files)}] wrote {p}")
        if args.html:
            p = write_html(book_id, text, layers, errors, args.out)
            print(f"[{i}/{len(files)}] wrote {p}")
        if args.opf:
            root = opf_root_for(Path(preprocessed["source_path"]))
            if root is None:
                print(f"  [warn] {book_id}: {preprocessed['source_path']} doesn't match "
                      f"<id>.opf/<id>.opf/base/v001.txt; skipping --opf output for this book",
                      file=sys.stderr)
            else:
                for p in write_opf_layers(layers, root):
                    print(f"[{i}/{len(files)}] wrote {p}")

        review_needed = {n: sum(1 for s in spans if s["confidence"] < args.review_threshold)
                         for n, spans in layers.items()}
        combined_errors = dict(errors)
        if preprocessed.get("errors"):
            combined_errors["preprocess"] = preprocessed["errors"]
        summary.append({
            "book_id": book_id, "text_length": text_length,
            "n_windows": predictions.get("n_windows", {}), "elapsed_s": predictions.get("elapsed_s", {}),
            "spans": {n: len(spans) for n, spans in layers.items()},
            "review_needed": review_needed,
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
        "books": summary,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {run_log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""postprocess.py -- Stage 3 of the layer-detection pipeline: rendering.

The command line and the per-book loop. The filters, review flags and the three
writers live in the sibling modules.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from ..layer_config import is_title_author
from .filters import (MIN_SPAN_CHARS, QUOTE_COVERAGE_DROP, drop_low_confidence, drop_non_tibetan_spans,
                      drop_short_spans, split_quoted_tsawa)
from .html_out import write_html
from .json_out import write_json
from .opf_out import (normalize_tsegs_tracked, opf_layers_to_write, opf_root_for, remap_layers,
                      write_opf_layers, write_plain_opf)
from .review import needs_review


def load_pair(book_id: str, predictions_dir: Path, preprocessed_dir: Path) -> tuple[dict, dict | None]:
    """Returns (predictions, preprocessed_or_None). preprocessed is None if
    Stage 1's file for this book_id is missing -- the caller should warn and
    skip JSON/HTML/opf output for that book, since none of them can be
    produced without at least the source text/length."""
    predictions = json.loads((predictions_dir / f"{book_id}.json").read_text(encoding="utf-8"))
    src_path = preprocessed_dir / f"{book_id}.json"
    preprocessed = json.loads(src_path.read_text(encoding="utf-8")) if src_path.is_file() else None
    return predictions, preprocessed


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
    ap.add_argument("--keep-low-confidence", action="store_true",
                    help="do not drop spans below a layer's min_confidence (Sabche: 0.85, see layer_config.py). "
                         "Default: drop them and list them in the batch summary.")
    ap.add_argument("--keep-quoted-tsawa", action="store_true",
                    help="do not drop tsawa spans that mostly overlap a predicted "
                         "quotation span. Default: drop when coverage >= 0.5 and "
                         "the quotation *model* produced spans. Does not use gold.")
    ap.add_argument("--title-author-review-threshold", type=float, default=None,
                    help="a title/author span with confidence below this is flagged review_needed "
                         "(wins over --review-threshold for those layers). Default: --review-threshold.")
    ap.add_argument("--title-author-opf", action="store_true",
                    help="with --opf, also write title and author as the OpenPecha layers BookTitle.yml and "
                         "Author.yml (the names the real books use). Off by default: --opf alone writes the "
                         "other layers only.")
    ap.add_argument("--review-threshold", type=float, default=0.7,
                    help="a span with confidence below this is flagged review_needed")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.title_author_opf and not args.opf:
        raise SystemExit("--title-author-opf only adds title/author to the .opf output: pass --opf as well")
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
        dropped_low_conf = {}
        if not args.keep_low_confidence:
            layers, dropped_low_conf = drop_low_confidence(layers)
            if dropped_low_conf:
                print(f"  [note] {book_id}: dropped spans below the layer's min confidence: "
                      f"{ {n: len(v) for n, v in dropped_low_conf.items()} }", file=sys.stderr)

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
            opf_layers = opf_layers_to_write(layers, args.title_author_opf)
            if not args.title_author_opf and any(is_title_author(n) and sp for n, sp in layers.items()):
                print(f"  [note] {book_id}: title/author are not written to .opf (add --title-author-opf)", file=sys.stderr)
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
            "low_confidence_dropped": {n: len(v) for n, v in dropped_low_conf.items()},
            "low_confidence_dropped_spans": dropped_low_conf,
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

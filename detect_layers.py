#!/usr/bin/env python3
"""detect_layers.py -- convenience wrapper: runs all three pipeline stages
in sequence, for users who just want one command.

The pipeline is split into three independently runnable stages, each its
own module with its own CLI:
    preprocess.py    text extraction        (.txt/.opf -> preprocessed/*.json)
    infer.py         model inference        (preprocessed/*.json -> predictions/*.json)
    postprocess.py   rendering + reporting  (predictions/*.json -> JSON/HTML/.opf + logs)

This file holds no detection logic of its own -- it builds the three
intermediate folders under --out and calls each stage's own main() in turn.
Run a stage directly (e.g. to re-render with a different --review-threshold
without re-running inference, or to resume a crashed batch via Stage 2's
checkpointing) instead of this wrapper when that's what you want.

Usage
-----
    # single document, all layers, both outputs
    python detect_layers.py --text book.txt --all --json --html --out results/

    # folder of books (renamed .txt files or raw OpenPecha layout), specific layers
    python detect_layers.py --dir books/ --layers tsawa sabche --json --out results/

    # write predictions back as OpenPecha layers too (layers/predicted/, not layers/v001/)
    python detect_layers.py --text data/raw_opf/P000201.opf/P000201.opf/base/v001.txt \
        --all --json --opf --out results/

Under --out, this creates:
    <out>/preprocessed/   Stage 1 output
    <out>/predictions/    Stage 2 output
    <out>/output/         Stage 3 output (JSON/HTML/.opf, batch summary, run log)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import infer  # noqa: E402
import postprocess  # noqa: E402
import preprocess  # noqa: E402
from layer_config import LAYERS  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", type=Path, help="a single text file")
    src.add_argument("--dir", type=Path, help="a folder of .txt files or OpenPecha book folders")
    ap.add_argument("--layers", nargs="+", choices=list(LAYERS), default=None)
    ap.add_argument("--all", action="store_true", help="run every layer with a model configured")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--html", action="store_true")
    ap.add_argument("--opf", action="store_true",
                    help="also write predictions as OpenPecha layer YAML under "
                         "layers/predicted/ (never layers/v001/); needs PyYAML")
    ap.add_argument("--review-threshold", type=float, default=0.7)
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--device", default=None, help="default: cuda if available, else cpu")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.layers and not args.all:
        raise SystemExit("pass --layers <name...> or --all")

    pre_dir = args.out / "preprocessed"
    pred_dir = args.out / "predictions"
    out_dir = args.out / "output"

    print(f"{'='*60}\nSTAGE 1: preprocess\n{'='*60}")
    preprocess.main(["--input", str(args.text or args.dir), "--out", str(pre_dir)])

    print(f"\n{'='*60}\nSTAGE 2: infer\n{'='*60}")
    infer_argv = ["--input", str(pre_dir), "--out", str(pred_dir)]
    infer_argv += ["--all"] if args.all else ["--layers", *args.layers]
    if args.device:
        infer_argv += ["--device", args.device]
    infer.main(infer_argv)

    print(f"\n{'='*60}\nSTAGE 3: postprocess\n{'='*60}")
    post_argv = ["--input", str(pred_dir), "--source", str(pre_dir), "--out", str(out_dir),
                "--review-threshold", str(args.review_threshold)]
    if args.json:
        post_argv.append("--json")
    if args.html:
        post_argv.append("--html")
    if args.opf:
        post_argv.append("--opf")
    postprocess.main(post_argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

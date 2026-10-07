#!/usr/bin/env python3
"""detect_layers.py -- convenience CLI: runs all three pipeline stages in
sequence, for users who just want one command (installed as `tibetan-detect`).

The pipeline is split into three independently runnable stages, each its
own module with its own console script:
    tibetan-preprocess    text extraction        (.txt/.opf -> preprocessed/*.json)
    tibetan-infer         model inference        (preprocessed/*.json -> predictions/*.json)
    tibetan-postprocess   rendering + reporting  (predictions/*.json -> JSON/HTML/.opf + logs)

This file holds no detection logic of its own -- it builds the three
intermediate folders under --out and calls each stage's own main() in turn.
For a Python API that returns results in-process instead of writing files
you manage yourself, see tibetan_layer_detection.detect() in __init__.py.

Usage
-----
    tibetan-detect --text book.txt --all --json --html --out results/
    tibetan-detect --dir books/ --layers tsawa sabche --json --out results/
    tibetan-detect --text data/raw_opf/P000201.opf/P000201.opf/base/v001.txt \\
        --all --json --opf --out results/

Under --out, this creates:
    <out>/preprocessed/   Stage 1 output
    <out>/predictions/    Stage 2 output
    <out>/output/         Stage 3 output (JSON/HTML/.opf, batch summary, run log)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import infer, postprocess, preprocess
from .layer_config import LAYERS


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
                         "layers/predicted/ (never layers/v001/)")
    ap.add_argument("--review-threshold", type=float, default=None,
                    help="flag a span review_needed below this confidence, for every layer. "
                         "Default: each layer's own threshold (see layer_config.py)")
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--device", default=None, help="default: cuda if available, else cpu")
    ap.add_argument("--stride", type=int, default=None,
                    help="override every selected layer's window stride; default keeps "
                         "each layer's own configured stride")
    ap.add_argument("--dual-window", action="store_true",
                    help="also run each layer at half window/stride size and flag spans "
                         "the two sizes disagree on; roughly doubles inference time")
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
    if args.stride is not None:
        infer_argv += ["--stride", str(args.stride)]
    if args.dual_window:
        infer_argv += ["--dual-window"]
    infer.main(infer_argv)

    print(f"\n{'='*60}\nSTAGE 3: postprocess\n{'='*60}")
    post_argv = ["--input", str(pred_dir), "--source", str(pre_dir), "--out", str(out_dir)]
    if args.review_threshold is not None:
        post_argv += ["--review-threshold", str(args.review_threshold)]
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

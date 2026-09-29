#!/usr/bin/env python3
"""preprocess.py -- Stage 1 of the layer-detection pipeline: text extraction.

Turns .txt files or raw OpenPecha .opf book folders into one plain JSON per
book, with no model involved. Splitting this out of detect_layers.py let it
run independently and be re-run without re-downloading/re-running any model.

Input can be a single .txt file, a single OpenPecha book folder (anything
containing a base/v001.txt at any depth, e.g. <id>.opf/ or the doubly-nested
<id>.opf/<id>.opf/), or a folder containing a mix of either -- discover_inputs()
combines a flat *.txt glob with a recursive **/base/v001.txt glob, which
covers all three cases without needing to special-case "is this an .opf
folder" at the top level. Same combined-discovery rule this repo's
detect_layers.py already used for --dir (commit efe60c0).

infer_book_id() is unchanged from detect_layers.py's existing version
(commit 6bb28fd / efe60c0): every OpenPecha book's text file is literally
named base/v001.txt, so the filename stem is useless as a book id -- it
uses the grandparent folder name instead (v001.txt's grandparent, e.g.
P000201.opf/P000201.opf/base/v001.txt -> P000201), stripping a trailing
".opf", and warns rather than guessing when that grandparent doesn't look
like a real OpenPecha id.

Usage
-----
    python preprocess.py --input book.txt --out preprocessed/
    python preprocess.py --input data/raw_opf/ --out preprocessed/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def discover_inputs(input_path: Path) -> list[Path]:
    """A .txt file -> itself. A directory -> every flat *.txt directly inside
    it, plus every base/v001.txt found at any depth."""
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise SystemExit(f"--input path does not exist: {input_path}")
    flat = set(input_path.glob("*.txt"))
    raw_opf = set(input_path.rglob("base/v001.txt"))
    return sorted(flat | raw_opf)


def infer_book_id(path: Path) -> str:
    """See module docstring."""
    if path.stem == "v001":
        grandparent = path.parent.parent.name
        if grandparent.endswith(".opf"):
            return grandparent[:-4]
        print(f"  [warn] {path}: named v001.txt but its grandparent folder "
              f"('{grandparent}') doesn't end in '.opf', so it doesn't look "
              f"like a real OpenPecha layout; using the generic book_id "
              f"'v001' instead of that folder name. This book may collide "
              f"with another v001.txt discovered in the same run.", file=sys.stderr)
    return path.stem


def load_text(path: Path) -> tuple[str | None, list[str]]:
    """Reads path as UTF-8. On a decode error, retries with errors="replace"
    and records a warning rather than failing outright -- skip (return None)
    only when there is no usable text at all (total read failure, or the
    result is empty)."""
    errors: list[str] = []
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as e:
        errors.append(f"utf-8 decode error ({e}); re-read with errors='replace'")
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as e2:
            return None, errors + [f"read failed after replace fallback: {e2}"]
    except OSError as e:
        return None, [f"read failed: {e}"]
    if not text:
        return None, errors + ["file is empty"]
    return text, errors


def preprocess_one(path: Path) -> dict | None:
    """Returns this book's JSON payload, or None if it was skipped (empty or
    unreadable -- a warning is already printed by the time this returns None)."""
    book_id = infer_book_id(path)
    text, errors = load_text(path)
    if text is None:
        print(f"  [warn] {book_id}: skipping ({'; '.join(errors)})", file=sys.stderr)
        return None
    return {
        "book_id": book_id,
        "source_path": str(path.resolve()),
        "text": text,
        "char_count": len(text),
        "encoding": "utf-8",
        "errors": errors,
    }


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", type=Path, required=True,
                    help=".txt file, OpenPecha book folder, or a folder of either")
    ap.add_argument("--out", type=Path, required=True)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    paths = discover_inputs(args.input)
    if not paths:
        raise SystemExit(f"no .txt or base/v001.txt files found under {args.input}")
    args.out.mkdir(parents=True, exist_ok=True)

    n_ok = n_skipped = 0
    for i, path in enumerate(paths, 1):
        payload = preprocess_one(path)
        if payload is None:
            n_skipped += 1
            continue
        out_path = args.out / f"{payload['book_id']}.json"
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{i}/{len(paths)}] wrote {out_path}  ({payload['char_count']:,} chars)")
        n_ok += 1

    print(f"\npreprocessed {n_ok} book(s), skipped {n_skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

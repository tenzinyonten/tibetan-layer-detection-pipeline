#!/usr/bin/env python3
"""infer.py -- Stage 2 of the layer-detection pipeline: model inference.

Reads Stage 1's preprocessed/<book_id>.json files, runs each requested
layer's model over the text (windowing, Viterbi decoding, confidence), and
writes predictions/<book_id>.json per book. No text rendering and no
filesystem writes back to any .opf here -- that's Stage 3's job; this stage
only ever produces raw span predictions.

Windowing (common/windows.py) and BIO/BIOE Viterbi decoding + confidence
(common/decode.py) are this repo's own extracted, minimal-dependency copies
(see those files) -- imported, not duplicated here.

Checkpointing: a book is skipped if predictions/<book_id>.json already
exists, so a crashed run resumes where it left off. This checks file
EXISTENCE only, not which layers that file covers -- if you change --layers
partway through a batch, delete the old predictions files for books you want
re-run with the new layer set; this script won't detect the mismatch itself.

Usage
-----
    python infer.py --input preprocessed/ --out predictions/ --all
    python infer.py --input preprocessed/ --out predictions/ --layers tsawa sabche --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from common.windows import pack_window_for_inference, sliding_windows, special_token_ids  # noqa: E402
from common.decode import (  # noqa: E402
    merge_char_spans, softmax, span_confidences, spans_from_bio, spans_from_bioe,
    transition_matrix, viterbi, viterbi_bioe,
)
from layer_config import LAYERS, LayerConfig  # noqa: E402

CONTENT_MAX = 8190  # max_length 8192 - CLS - SEP, shared by every layer here


def _window_logits(text_input_ids, offsets, w_start, w_end, cls_id, sep_id, pad_id, model, device):
    """Run one window through the model. Returns logits for its content tokens only."""
    import torch

    ids, mask, content_off = pack_window_for_inference(
        text_input_ids, offsets, w_start, w_end, cls_id, sep_id, pad_id, max_length=8192)
    out = model(input_ids=torch.tensor([ids], device=device),
               attention_mask=torch.tensor([mask], device=device)).logits[0]
    logits = out.float().cpu().numpy()
    n_content = len(content_off)
    return logits[1:1 + n_content], content_off


def run_layer(text: str, tok, model, cfg: LayerConfig, device: str) -> tuple[list[tuple[int, int, float]], int]:
    """Returns (character spans as (start, end, confidence) triples, number of windows run).

    Two decode policies, per cfg.stitch_first_window:
      - default (Tsawa, Sabche, Chapter, Quotation): decode each window
        independently, then union-merge the resulting character spans. This
        is what each of those models' own published test scores used.
      - stitched (Yigchung only): a token's logits come only from the first
        window that covers it (matching how later copies were masked out of
        the training loss), and the whole document is Viterbi-decoded once
        as a single sequence. See run_layer_stitched().
    """
    import torch

    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = enc["input_ids"]
    offsets = [tuple(o) for o in enc["offset_mapping"]]
    cls_id, sep_id, pad_id = special_token_ids(tok)
    wins = sliding_windows(len(input_ids), CONTENT_MAX, cfg.stride)
    model.eval()

    if cfg.stitch_first_window:
        with torch.no_grad():
            spans = run_layer_stitched(input_ids, offsets, wins, cls_id, sep_id, pad_id, model, cfg, device)
        return spans, len(wins)

    all_spans: list[tuple[int, int, float]] = []
    with torch.no_grad():
        for w_start, w_end in wins:
            content_logits, content_off = _window_logits(
                input_ids, offsets, w_start, w_end, cls_id, sep_id, pad_id, model, device)
            n_content = len(content_off)
            if cfg.scheme == "bio":
                seq = viterbi(content_logits, cfg.break_penalty)
                tok_spans = spans_from_bio(seq)
            else:
                seq = viterbi_bioe(content_logits, cfg.break_penalty)
                tok_spans = spans_from_bioe(seq)
            probs = softmax(content_logits)
            confs = span_confidences(probs, seq, tok_spans)
            for (a, b), conf in zip(tok_spans, confs):
                if a >= n_content or b >= n_content:
                    continue
                cs, ce = content_off[a][0], content_off[b][1]
                if ce > cs:
                    all_spans.append((cs, ce, conf))
    return merge_char_spans(all_spans), len(wins)


def run_layer_stitched(input_ids, offsets, wins, cls_id, sep_id, pad_id, model, cfg: LayerConfig, device) -> list[tuple[int, int, float]]:
    """Yigchung's own inference recipe: each absolute token index gets its
    logits from the FIRST window (in window order) that covers it -- this
    matches training, where a token repeated in a later, overlapping window
    was masked out of the loss (-100). The stitched, whole-document logit
    sequence is then Viterbi-decoded once, not per window then merged."""
    n_tokens = len(input_ids)
    n_labels = 4 if cfg.scheme == "bioe" else 3
    stitched = np.zeros((n_tokens, n_labels), dtype=np.float32)
    owned = np.zeros(n_tokens, dtype=bool)
    for w_start, w_end in wins:
        content_logits, content_off = _window_logits(
            input_ids, offsets, w_start, w_end, cls_id, sep_id, pad_id, model, device)
        for i, abs_idx in enumerate(range(w_start, w_end)):
            if not owned[abs_idx]:
                stitched[abs_idx] = content_logits[i]
                owned[abs_idx] = True
    seq = viterbi(stitched, cfg.break_penalty) if cfg.scheme == "bio" else viterbi_bioe(stitched, cfg.break_penalty)
    tok_spans = spans_from_bio(seq) if cfg.scheme == "bio" else spans_from_bioe(seq)
    probs = softmax(stitched)
    confs = span_confidences(probs, seq, tok_spans)
    out = []
    for (a, b), conf in zip(tok_spans, confs):
        cs, ce = offsets[a][0], offsets[b][1]
        if ce > cs:
            out.append((cs, ce, conf))
    return out


# ---------------------------------------------------------------------------
# model loading
# ---------------------------------------------------------------------------

_MODEL_CACHE: dict[str, tuple] = {}


def load_layer_model(name: str, device: str):
    """Returns (tokenizer, model) or raises. Cached across books in one run."""
    if name in _MODEL_CACHE:
        return _MODEL_CACHE[name]
    cfg = LAYERS[name]
    if cfg.repo is None:
        raise RuntimeError(f"no model configured for layer '{name}' ({cfg.source})")
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    token = os.environ.get("HF_TOKEN")
    try:
        tok = AutoTokenizer.from_pretrained(cfg.repo, token=token)
    except Exception as e:
        # BDRC/Bo-Quotation-Detection and BDRC/Bo-Yigchung-Detection ship a
        # tokenizer_config.json in a newer format (extra_special_tokens as a
        # list) that this transformers version cannot parse. Their own
        # READMEs and the Sabche/Tsawa/Chapter cards all state the tokenizer
        # is an unchanged copy of jhu-clsp/mmBERT-base, so falling back to
        # that base copy is not a behavior change, only a workaround.
        print(f"  [warn] {cfg.repo}: own tokenizer failed to load ({e}); "
              f"falling back to jhu-clsp/mmBERT-base", file=sys.stderr)
        tok = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-base", token=token)
    if not tok.is_fast:
        raise RuntimeError(f"{cfg.repo}: tokenizer is not a fast tokenizer (need offset mapping)")
    tok.model_max_length = int(1e12)
    model = AutoModelForTokenClassification.from_pretrained(cfg.repo, token=token)
    n_expected = 3 if cfg.scheme == "bio" else 4
    if model.config.num_labels != n_expected:
        raise RuntimeError(
            f"{cfg.repo}: expected {n_expected} labels for scheme '{cfg.scheme}', "
            f"got {model.config.num_labels}")
    model.to(device)
    if cfg.caveat:
        print(f"  [note] {name}: {cfg.caveat}", file=sys.stderr)
    _MODEL_CACHE[name] = (tok, model)
    return tok, model


# ---------------------------------------------------------------------------
# per-book inference
# ---------------------------------------------------------------------------

def already_done(book_id: str, out_dir: Path) -> bool:
    """Checkpointing: True if out_dir/<book_id>.json already exists. File
    existence only -- does not check which layers that file covers."""
    return (out_dir / f"{book_id}.json").exists()


def infer_one(book: dict, layer_names: list[str], device: str) -> dict:
    """book is one Stage 1 payload (book_id + text, at least). Returns the
    predictions payload: raw (unrounded) confidence per span, plus n_windows
    and elapsed_s per layer and any per-layer errors, so Stage 3 can report
    timing/window counts without re-running inference."""
    text, book_id = book["text"], book["book_id"]
    layers: dict[str, list[dict]] = {}
    n_windows: dict[str, int] = {}
    elapsed_s: dict[str, float] = {}
    errors: dict[str, str] = {}
    for name in layer_names:
        t0 = time.time()
        try:
            tok, model = load_layer_model(name, device)
        except Exception as e:  # missing model / load failure: skip and warn
            errors[name] = f"model load failed: {e}"
            print(f"  [warn] {book_id}: layer '{name}' skipped: {e}", file=sys.stderr)
            continue
        try:
            spans, n_win = run_layer(text, tok, model, LAYERS[name], device)
            layers[name] = [{"start": s, "end": e, "confidence": c} for s, e, c in spans]
            n_windows[name] = n_win
        except Exception as e:
            errors[name] = f"inference failed: {e}"
            print(f"  [warn] {book_id}: layer '{name}' failed: {e}", file=sys.stderr)
        elapsed_s[name] = round(time.time() - t0, 2)
    return {"book_id": book_id, "layers": layers, "n_windows": n_windows,
            "elapsed_s": elapsed_s, "errors": errors}


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", type=Path, required=True, help="folder of Stage 1 preprocessed JSONs")
    ap.add_argument("--out", type=Path, required=True, help="folder to write predictions JSONs into")
    ap.add_argument("--layers", nargs="+", choices=list(LAYERS), default=None)
    ap.add_argument("--all", action="store_true", help="run every layer with a model configured")
    ap.add_argument("--device", default="cuda" if _has_cuda() else "cpu")
    return ap.parse_args(argv)


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.layers and not args.all:
        raise SystemExit("pass --layers <name...> or --all")
    layer_names = list(LAYERS) if args.all else args.layers

    files = sorted(args.input.glob("*.json"))
    if not files:
        raise SystemExit(f"no preprocessed JSON files found in {args.input}")
    args.out.mkdir(parents=True, exist_ok=True)

    n_done = n_skipped = 0
    for i, fp in enumerate(files, 1):
        book = json.loads(fp.read_text(encoding="utf-8"))
        book_id = book["book_id"]
        if already_done(book_id, args.out):
            print(f"[{i}/{len(files)}] {book_id}: already has predictions, skipping (checkpoint)")
            n_skipped += 1
            continue

        t0 = time.time()
        result = infer_one(book, layer_names, args.device)
        elapsed = time.time() - t0
        out_path = args.out / f"{book_id}.json"
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

        counts = {n: len(result["layers"].get(n, [])) for n in layer_names}
        print(f"[{i}/{len(files)}] {book_id}  ({elapsed:.1f}s)  spans={counts}"
              + (f"  errors={result['errors']}" if result["errors"] else ""))
        n_done += 1

    print(f"\ninferred {n_done} book(s), skipped {n_skipped} already done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

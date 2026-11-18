#!/usr/bin/env python3
"""infer.py -- Stage 2 of the layer-detection pipeline: model inference.

Reads Stage 1's preprocessed/<book_id>.json files, runs each requested
layer's model over the text (windowing, Viterbi decoding, confidence), and
writes predictions/<book_id>.json per book.

Windowing (.windows) and BIO/BIOE Viterbi decoding + confidence (.decode)
are self-contained, dependency-light copies of the logic in this repo's
common/build_tsawa_dataset.py and common/eval_viterbi_iou.py -- imported as
package-relative modules here (not via a sys.path hack to a sibling
directory) so this package is actually importable once pip-installed
(site-packages only contains the package's own files, not this repo's
top-level common/).

Checkpointing: a book is skipped if predictions/<book_id>.json already
exists, so a crashed run resumes where it left off. This checks file
EXISTENCE only, not which layers that file covers.

Usage (as an installed console script)
---------------------------------------
    tibetan-infer --input preprocessed/ --out predictions/ --all
    tibetan-infer --input preprocessed/ --out predictions/ --layers tsawa sabche --device cuda
    tibetan-infer --input preprocessed/ --out predictions/ --all --workers 3   # 3 books at a time

--workers N runs N books at the same time, each in its own process. Every process loads its
own copy of the models (about 5.5 GB of RAM for all seven layers), so N books need about
N x 5.5 GB. The default, 1, is the plain one-book-at-a-time loop.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from .decode import (
    MIN_SPAN_CHARS, apply_linguistic_rules, merge_dual_window, merge_spans_with_votes,
    overlap_regions, softmax, span_confidences, spans_from_bio, spans_from_bioe,
    viterbi, viterbi_bioe,
)
from .layer_config import AUTHOR_MODEL, LAYERS, TITLE_MODEL, LayerConfig, TitleAuthorOptions
from .windows import pack_window_for_inference, sliding_windows, special_token_ids

CONTENT_MAX = 8190  # max_length 8192 - CLS - SEP, shared by every layer here


def _window_logits(text_input_ids, offsets, w_start, w_end, cls_id, sep_id, pad_id, model, device,
                   max_length=8192):
    """Run one window through the model. Returns logits for its content tokens only.

    max_length: CLS + content + SEP (+ padding); defaults to the shared 8192
    every layer trained at. --dual-window's half-size pass calls this with a
    smaller max_length instead."""
    import torch

    ids, mask, content_off = pack_window_for_inference(
        text_input_ids, offsets, w_start, w_end, cls_id, sep_id, pad_id, max_length=max_length)
    out = model(input_ids=torch.tensor([ids], device=device),
               attention_mask=torch.tensor([mask], device=device)).logits[0]
    logits = out.float().cpu().numpy()
    n_content = len(content_off)
    return logits[1:1 + n_content], content_off


def run_layer(text: str, tok, model, cfg: LayerConfig, device: str,
             stride: int | None = None, content_max: int = CONTENT_MAX,
             ) -> tuple[list[tuple[int, int, float, int]], int]:
    """Returns (character spans as (start, end, confidence, window_votes)
    4-tuples, number of windows run).

    stride: overrides cfg.stride for this call (e.g. from --stride). None
        (the default) keeps each layer's own configured stride -- tsawa/
        sabche/chapter at 5120, quotation at 3613, yigchung at 4914 -- so
        omitting --stride is unchanged from before this option existed.
    content_max: window content size in tokens (CLS/SEP are added on top).
        Defaults to the shared CONTENT_MAX every layer trained at; passed
        smaller by run_layer_dual's half-size pass (--dual-window).

    window_votes is how many distinct windows' predictions were merged into
    that span; confidence is adjusted by that count (see
    decode.merge_spans_with_votes). Always 1 for Yigchung (stitch_first_window
    -- see run_layer_stitched), since it decodes the whole document once from
    a stitched logit sequence rather than merging per-window predictions, so
    "which window voted for this span" does not apply there.

    Two decode policies, per cfg.stitch_first_window:
      - default (Tsawa, Sabche, Chapter, Quotation): decode each window
        independently, then union-merge the resulting character spans.
      - stitched (Yigchung only): a token's logits come only from the first
        window that covers it, and the whole document is Viterbi-decoded
        once as a single sequence. See run_layer_stitched().
    """
    import torch

    eff_stride = cfg.stride if stride is None else stride
    max_length = content_max + 2  # + CLS + SEP
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = enc["input_ids"]
    offsets = [tuple(o) for o in enc["offset_mapping"]]
    cls_id, sep_id, pad_id = special_token_ids(tok)
    wins = sliding_windows(len(input_ids), content_max, eff_stride)
    model.eval()

    if cfg.stitch_first_window:
        with torch.no_grad():
            spans = run_layer_stitched(input_ids, offsets, wins, cls_id, sep_id, pad_id, model, cfg,
                                       device, max_length=max_length)
        return spans, len(wins)

    all_spans: list[tuple[int, int, float, int]] = []
    window_char_ranges: list[tuple[int, int]] = []
    with torch.no_grad():
        for widx, (w_start, w_end) in enumerate(wins):
            content_logits, content_off = _window_logits(
                input_ids, offsets, w_start, w_end, cls_id, sep_id, pad_id, model, device,
                max_length=max_length)
            window_char_ranges.append((content_off[0][0], content_off[-1][1]))
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
                    all_spans.append((cs, ce, conf, widx))
    regions = overlap_regions(window_char_ranges)
    return merge_spans_with_votes(all_spans, regions), len(wins)


def run_layer_stitched(input_ids, offsets, wins, cls_id, sep_id, pad_id, model, cfg: LayerConfig, device,
                       max_length=8192) -> list[tuple[int, int, float, int]]:
    """Yigchung's own inference recipe: each absolute token index gets its
    logits from the FIRST window that covers it, matching how later copies
    were masked out of the training loss. The stitched, whole-document logit
    sequence is then Viterbi-decoded once, not per window then merged."""
    n_tokens = len(input_ids)
    n_labels = 4 if cfg.scheme == "bioe" else 3
    stitched = np.zeros((n_tokens, n_labels), dtype=np.float32)
    owned = np.zeros(n_tokens, dtype=bool)
    for w_start, w_end in wins:
        content_logits, content_off = _window_logits(
            input_ids, offsets, w_start, w_end, cls_id, sep_id, pad_id, model, device,
            max_length=max_length)
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
            out.append((cs, ce, conf, 1))  # window_votes: not meaningful here, see run_layer
    return out


def rule_flags(cfg: LayerConfig) -> dict:
    """The layer's cleanup switches as apply_linguistic_rules keyword arguments."""
    return {"extend_to_tsheg": cfg.extend_to_tsheg, "merge_gaps": cfg.merge_small_gaps,
            "repair": cfg.repair_fragments,
            "extend_fragments": cfg.extend_lone_fragments, "shad_rule": cfg.shad_rule}


def run_layer_dual(text: str, tok, model, cfg: LayerConfig, device: str,
                   stride: int | None = None,
                   ) -> tuple[list[tuple[int, int, float, int, str]], int, int]:
    """--dual-window: runs the layer twice, once at the normal (possibly
    --stride-overridden) window size and once at half window size / half
    stride, each independently cleaned up by apply_linguistic_rules, then
    reconciled by decode.merge_dual_window. Returns (spans as 5-tuples with a
    trailing dual_window_agreement string, total windows run across both
    passes, total short spans removed across both passes)."""
    full_stride = cfg.stride if stride is None else stride
    full_spans, n_full = run_layer(text, tok, model, cfg, device, stride=full_stride,
                                   content_max=CONTENT_MAX)
    full_spans, n_removed_full = apply_linguistic_rules(full_spans, text, **rule_flags(cfg))

    half_stride = max(1, full_stride // 2)
    half_spans, n_half = run_layer(text, tok, model, cfg, device, stride=half_stride,
                                   content_max=CONTENT_MAX // 2)
    half_spans, n_removed_half = apply_linguistic_rules(half_spans, text, **rule_flags(cfg))

    merged = merge_dual_window(full_spans, half_spans)
    return merged, n_full + n_half, n_removed_full + n_removed_half


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
        # list) that some transformers versions cannot parse. Their own
        # READMEs state the tokenizer is an unchanged copy of
        # jhu-clsp/mmBERT-base, so falling back to that base copy is not a
        # behavior change, only a workaround.
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


def infer_one(book: dict, layer_names: list[str], device: str,
             stride: int | None = None, dual_window: bool = False,
             ta: TitleAuthorOptions | None = None) -> dict:
    """book is one Stage 1 payload (book_id + text, at least). Returns the
    predictions payload: raw (unrounded) confidence per span, plus n_windows,
    elapsed_s and short_spans_removed per layer and any per-layer errors.

    stride: see run_layer -- None keeps each layer's own configured stride.
    dual_window: see run_layer_dual -- doubles inference time per layer when
        True; each span gets an extra "dual_window_agreement" field.
    ta: how the title/author layers run (see layer_config.TitleAuthorOptions);
        stride and dual_window do not apply to them."""
    text, book_id = book["text"], book["book_id"]
    layers: dict[str, list[dict]] = {}
    n_windows: dict[str, int] = {}
    elapsed_s: dict[str, float] = {}
    short_spans_removed: dict[str, int] = {}
    errors: dict[str, str] = {}
    for name in layer_names:
        if LAYERS[name].kind == "head_tail":
            continue  # title/author: handled together below, one forward pass for both fields
        t0 = time.time()
        try:
            tok, model = load_layer_model(name, device)
        except Exception as e:  # missing model / load failure: skip and warn
            errors[name] = f"model load failed: {e}"
            print(f"  [warn] {book_id}: layer '{name}' skipped: {e}", file=sys.stderr)
            continue
        try:
            if dual_window:
                spans5, n_win, n_removed = run_layer_dual(text, tok, model, LAYERS[name], device,
                                                           stride=stride)
                layers[name] = [{"start": s, "end": e, "confidence": c, "window_votes": v,
                                "dual_window_agreement": agreement}
                               for s, e, c, v, agreement in spans5]
            else:
                spans, n_win = run_layer(text, tok, model, LAYERS[name], device, stride=stride)
                spans, n_removed = apply_linguistic_rules(spans, text, **rule_flags(LAYERS[name]))
                layers[name] = [{"start": s, "end": e, "confidence": c, "window_votes": v}
                               for s, e, c, v in spans]
            if n_removed:
                print(f"  [note] {book_id}: layer '{name}': removed {n_removed} span(s) "
                      f"(under {MIN_SPAN_CHARS} chars, duplicate fragments, or no Tibetan text)", file=sys.stderr)
            short_spans_removed[name] = n_removed
            n_windows[name] = n_win
        except Exception as e:
            errors[name] = f"inference failed: {e}"
            print(f"  [warn] {book_id}: layer '{name}' failed: {e}", file=sys.stderr)
        elapsed_s[name] = round(time.time() - t0, 2)
    ta_fields = [n for n in layer_names if LAYERS[n].kind == "head_tail"]
    if ta_fields:
        from .title_author import run_title_author
        ta_layers, ta_elapsed, ta_errors = run_title_author(text, ta_fields, ta or TitleAuthorOptions(), device)
        layers.update(ta_layers)
        elapsed_s.update(ta_elapsed)
        errors.update(ta_errors)
        n_windows.update({n: 1 for n in ta_layers})
    return {"book_id": book_id, "layers": layers, "n_windows": n_windows,
            "elapsed_s": elapsed_s, "short_spans_removed": short_spans_removed, "errors": errors}


def layers_for_run(requested: list[str], *, skip_quotation: bool = False) -> list[str]:
    """Ensure the quotation *model* runs whenever tsawa does, so postprocess
    can drop tsawa spans that sit inside predicted quotations."""
    names = list(requested)
    if "tsawa" in names and "quotation" not in names and not skip_quotation:
        names.append("quotation")
    return names


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", type=Path, required=True, help="folder of Stage 1 preprocessed JSONs")
    ap.add_argument("--out", type=Path, required=True, help="folder to write predictions JSONs into")
    ap.add_argument("--layers", nargs="+", choices=list(LAYERS), default=None)
    ap.add_argument("--all", action="store_true", help="run every layer with a model configured")
    ap.add_argument("--author-span", choices=["last", "first"], default="last",
                    help="which author span to keep when the author model finds several. Default last "
                         "(the rule for a single work). For whole books the gold author is usually on the "
                         "title page, and first matched it about twice as often (35%% against 17%%, 511 books).")
    ap.add_argument("--title-model", default=TITLE_MODEL,
                    help="Hugging Face repo (or local folder) of the model that finds the title. "
                         "Default: the title-only model (3 labels, Viterbi penalty 6). The joint "
                         "Yontenn/mmbert-title-author-run2 also works (penalty 12).")
    ap.add_argument("--author-model", default=AUTHOR_MODEL,
                    help="Hugging Face repo (or local folder) of the model that finds the author. "
                         "Default: the author-only model. If --title-model and --author-model are the "
                         "same joint model, one forward pass serves both fields.")
    ap.add_argument("--device", default="cuda" if _has_cuda() else "cpu")
    ap.add_argument("--stride", type=int, default=None,
                    help="override every selected layer's window stride (smaller = more "
                         "overlap between windows). Default: each layer keeps its own "
                         "configured stride (tsawa/sabche/chapter 5120, quotation 3613, "
                         "yigchung 4914) -- unchanged from before this option existed.")
    ap.add_argument("--skip-quotation", action="store_true",
                    help="do not auto-run the quotation model when tsawa is selected. "
                         "Default: tsawa also runs Bo-Quotation-Detection so Stage 3 can "
                         "drop quoted tsawa (no gold layers needed).")
    ap.add_argument("--dual-window", action="store_true",
                    help="also run each layer at half window/stride size and keep only "
                         "spans both runs agree on at high confidence; spans only one run "
                         "found are flagged dual_window_agreement for review. Roughly "
                         "doubles inference time per layer.")
    ap.add_argument("--workers", type=int, default=1,
                    help="how many books to process at the same time, each in its own process. Default 1 "
                         "(one book at a time). Every process loads its own copy of the models, about "
                         "5.5 GB of RAM for all seven layers, so N workers need about N x 5.5 GB.")
    ap.add_argument("--threads-per-worker", type=int, default=None,
                    help="PyTorch threads in each worker process. Default: PyTorch's own default divided "
                         "by --workers (at least 1), so the workers share the same cores.")
    return ap.parse_args(argv)


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _write_prediction(out_path: Path, result: dict) -> None:
    """Write a predictions file so that a crash never leaves a half-written one (the checkpoint
    treats an existing file as done): write to a temporary name, then rename."""
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, out_path)


def _init_worker(threads: int | None) -> None:
    if threads:
        import torch
        torch.set_num_threads(threads)


def _infer_file(job: tuple) -> tuple:
    """One book in a worker process: returns (book_id, elapsed_s, counts, errors)."""
    fp, out_dir, layer_names, device, stride, dual_window, ta = job
    book = json.loads(Path(fp).read_text(encoding="utf-8"))
    t0 = time.time()
    result = infer_one(book, layer_names, device, stride=stride, dual_window=dual_window, ta=ta)
    elapsed = time.time() - t0
    _write_prediction(Path(out_dir) / f"{book['book_id']}.json", result)
    counts = {n: len(result["layers"].get(n, [])) for n in layer_names}
    return book["book_id"], elapsed, counts, result["errors"]


def run_in_pool(fn, jobs: list, workers: int, threads: int | None = None, start_method: str = "spawn"):
    """Run fn(job) for every job in `workers` processes; yields (job, result, error) as each one
    finishes. One failing job gives (job, None, error) and the others carry on. Each process
    sets PyTorch to `threads` threads (None: leave PyTorch's default). "spawn" is used because
    PyTorch (and CUDA) are not safe to fork."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed
    ctx = multiprocessing.get_context(start_method)
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_init_worker,
                             initargs=(threads,)) as ex:
        futures = {ex.submit(fn, job): job for job in jobs}
        for fut in as_completed(futures):
            try:
                yield futures[fut], fut.result(), None
            except Exception as e:  # one book failing must not stop the batch
                yield futures[fut], None, e


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.layers and not args.all:
        raise SystemExit("pass --layers <name...> or --all")
    layer_names = layers_for_run(
        list(LAYERS) if args.all else list(args.layers),
        skip_quotation=args.skip_quotation,
    )

    files = sorted(args.input.glob("*.json"))
    if not files:
        raise SystemExit(f"no preprocessed JSON files found in {args.input}")
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        ta_options = TitleAuthorOptions(
            author_span=args.author_span,
            title_model=args.title_model, author_model=args.author_model)
    except ValueError as e:
        raise SystemExit(str(e))

    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    n_done = n_skipped = n_failed = 0
    if args.workers == 1:
        for i, fp in enumerate(files, 1):
            book = json.loads(fp.read_text(encoding="utf-8"))
            book_id = book["book_id"]
            if already_done(book_id, args.out):
                print(f"[{i}/{len(files)}] {book_id}: already has predictions, skipping (checkpoint)")
                n_skipped += 1
                continue

            t0 = time.time()
            result = infer_one(book, layer_names, args.device, stride=args.stride,
                              dual_window=args.dual_window, ta=ta_options)
            elapsed = time.time() - t0
            _write_prediction(args.out / f"{book_id}.json", result)

            counts = {n: len(result["layers"].get(n, [])) for n in layer_names}
            print(f"[{i}/{len(files)}] {book_id}  ({elapsed:.1f}s)  spans={counts}"
                  + (f"  errors={result['errors']}" if result["errors"] else ""))
            n_done += 1
    else:
        pending = []
        for fp in files:
            if already_done(fp.stem, args.out):
                n_skipped += 1
            else:
                pending.append(fp)
        if n_skipped:
            print(f"{n_skipped} book(s) already have predictions, skipping (checkpoint)")
        workers = min(args.workers, len(pending)) if pending else 0
        if workers:
            threads = args.threads_per_worker
            if threads is None:
                import torch
                threads = max(1, torch.get_num_threads() // workers)
            if args.device.startswith("cuda"):
                print(f"  [note] {workers} workers share one GPU: each loads its own copy of the models "
                      f"onto it", file=sys.stderr)
            print(f"running {len(pending)} book(s) in {workers} worker process(es), {threads} thread(s) each")
            jobs = [(fp, args.out, layer_names, args.device, args.stride, args.dual_window, ta_options)
                    for fp in pending]
            for k, (job, res, err) in enumerate(run_in_pool(_infer_file, jobs, workers, threads), 1):
                if err is not None:
                    print(f"[{k}/{len(pending)}] {job[0].stem}  FAILED: {err!r}", file=sys.stderr)
                    n_failed += 1
                    continue
                book_id, elapsed, counts, errors = res
                print(f"[{k}/{len(pending)}] {book_id}  ({elapsed:.1f}s)  spans={counts}"
                      + (f"  errors={errors}" if errors else ""))
                n_done += 1

    print(f"\ninferred {n_done} book(s), skipped {n_skipped} already done"
          + (f", {n_failed} FAILED" if n_failed else ""))
    return 1 if n_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

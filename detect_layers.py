#!/usr/bin/env python3
"""detect_layers.py — run one or more layer-detection models over Tibetan text(s).

Loads a per-layer mmBERT token-classification model, slides 8192-token windows
over the input text (each layer's own stride, see LAYERS below), Viterbi-decodes
each window (BIO layers: 3-state; Quotation: 4-state BIOE), maps token spans to
character offsets, and merges predictions from overlapping windows.

Reuses windowing from common/build_tsawa_dataset.py (sliding_windows,
special_token_ids) and the BIO Viterbi decoder from common/eval_viterbi_iou.py
(transition_matrix, viterbi, spans_from_bio). The BIOE decoder for Quotation is
new: no BIOE Viterbi existed anywhere in this repo before this file. The
raw-text -> windows -> model -> character-offset path is also new; the closest
prior art was chapter/src/analyze_chapter_errors.py, written for scoring
against gold spans, not for a standalone detection API.

Per-layer settings (break penalty, stride) come from each model's own
Hugging Face README / training/test_metrics.json / training/config.yaml,
not from a single project-wide default -- see LAYERS below for the source of
each number. All five layers (BDRC/Bo-Tsawa-Detection, Bo-Sabche-detection,
Bo-Chapter-Detection, Bo-Quotation-Detection, Bo-Yigchung-Detection) have a
dedicated model as of 2026-09-28. Yigchung uses a different windowing stride
(4914, not 5120) and a different overlap policy: a token's logits come from
only the first window that covers it, and the whole document is decoded once
(see run_layer_stitched()), matching how its training data masked repeated
tokens out of the loss. Its own README also says to cite a test F1 of 0.431,
well below the other four layers; that caveat is printed whenever it loads.
BDRC/Bo-Multilayer-Detection (the old 15-label joint model) is deliberately
not used for anything here: its own published metrics put tsawa/yigchung/
quote F1 at about 0.02-0.04.

Usage
-----
    # single document, all available layers, both outputs
    python common/detect_layers.py --text book.txt --all --json --html --out results/

    # folder of books, specific layers only
    python common/detect_layers.py --dir books/ --layers tsawa sabche --json --out results/

    # single document, quick check, print to terminal
    python common/detect_layers.py --text book.txt --all
"""

from __future__ import annotations

import argparse
import html as html_mod
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from build_tsawa_dataset import sliding_windows, special_token_ids  # noqa: E402
from eval_viterbi_iou import spans_from_bio, transition_matrix, viterbi  # noqa: E402

CONTENT_MAX = 8190  # max_length 8192 - CLS - SEP, shared by every layer here


@dataclass
class LayerConfig:
    repo: str | None
    scheme: str  # "bio" or "bioe"
    break_penalty: float
    stride: int
    color: str
    source: str  # where break_penalty/stride came from, for the --help text and logs
    stitch_first_window: bool = False  # see run_layer_stitched(); only Yigchung as of 2026-09-28
    caveat: str = ""  # printed once when the layer is loaded, for a known-weak model


# Each break_penalty/stride is the value documented on the model's own Hugging
# Face repo (README.md / training/test_metrics.json / training/config.yaml),
# checked 2026-09-28, not an assumed project-wide default.
LAYERS: dict[str, LayerConfig] = {
    "tsawa": LayerConfig(
        repo="BDRC/Bo-Tsawa-Detection", scheme="bio",
        break_penalty=4.0, stride=5120, color="#f5e34d",  # yellow
        source="BDRC/Bo-Tsawa-Detection training/test_metrics.json (decoding.break_penalty)",
    ),
    "sabche": LayerConfig(
        repo="BDRC/Bo-Sabche-detection", scheme="bio",  # NB: repo has lowercase 'detection'
        break_penalty=4.0, stride=5120, color="#4d9df5",  # blue
        source="BDRC/Bo-Sabche-detection training/test_metrics.json (decoding.break_penalty)",
    ),
    "chapter": LayerConfig(
        repo="BDRC/Bo-Chapter-Detection", scheme="bio",
        break_penalty=4.0, stride=5120, color="#4df57a",  # green
        source="BDRC/Bo-Chapter-Detection training/test_metrics.json (decoding.break_penalty); "
               "dataset revision c8c758ed matches the stratified Chapter split built in this repo",
    ),
    "quotation": LayerConfig(
        repo="BDRC/Bo-Quotation-Detection", scheme="bioe",
        break_penalty=0.0, stride=3613, color="#f5a23d",  # orange
        source="BDRC/Bo-Quotation-Detection training/config.yaml / training/test_metrics.json "
               "(decode_chosen_on_validation.break_penalty; max_length 8192 / stride 3613)",
    ),
    "yigchung": LayerConfig(
        repo="BDRC/Bo-Yigchung-Detection", scheme="bio",
        break_penalty=1.0, stride=4914, color="#f57ec0",  # pink
        stitch_first_window=True,
        source="BDRC/Bo-Yigchung-Detection README.md (break_penalty 1, stride 4914 from "
               "the 'windowed_w8192_s4914' config; a token's logits come from only the "
               "FIRST window that covers it, matching how overlap was masked at train time "
               "-- decode once over the whole stitched document, not per window then merged)",
        caveat="this model's own README says to cite its test F1 of 0.431 (precision 0.464, "
               "recall 0.402), well below the other layers; one test book had 107 gold spans "
               "and zero true positives at this break penalty",
    ),
}


# ---------------------------------------------------------------------------
# BIOE Viterbi (new: no BIOE decoder existed in common/ before this file)
# ---------------------------------------------------------------------------
# Label ids, matching karma689/tibetan-quotation-detection's config.json and
# train_layer.py's "bioe" scheme (ENTITIES["bioe"] = (b=1, i=2, e=3)):
#   O=0  B=1  I=2  E=3
# Legal grammar: a span is exactly  B I* E  (an I never opens or closes a
# span, and O never continues one). First token may not be I or E; a token
# after E may be O (span closed) or a fresh B (new span starts immediately).
NEG = -1.0e9
O, B, I, E = 0, 1, 2, 3


def bioe_transition_matrix(break_penalty: float) -> np.ndarray:
    """m[i, j] = cost of moving FROM state i TO state j (added to j's score)."""
    m = np.zeros((4, 4))
    legal = {
        (O, O), (O, B),
        (B, I), (B, E),
        (I, I), (I, E),
        (E, O), (E, B),
    }
    for i in range(4):
        for j in range(4):
            if (i, j) not in legal:
                m[i, j] = NEG
    m[E, O] -= break_penalty  # the only legal way to end a span
    return m


def viterbi_bioe(logits: np.ndarray, break_penalty: float) -> np.ndarray:
    T, C = logits.shape
    if C != 4:
        raise SystemExit(f"viterbi_bioe expects 4 labels, got {C}")
    tr = bioe_transition_matrix(break_penalty)
    dp = np.full((T, C), NEG)
    back = np.zeros((T, C), dtype=np.int64)
    dp[0] = logits[0]
    dp[0, I] = NEG
    dp[0, E] = NEG
    for t in range(1, T):
        s = dp[t - 1][:, None] + tr
        back[t] = s.argmax(0)
        dp[t] = s.max(0) + logits[t]
    # the sequence must not end mid-span (B or I with no E)
    dp[-1, B] = NEG
    dp[-1, I] = NEG
    path = np.zeros(T, dtype=np.int64)
    path[-1] = int(dp[-1].argmax())
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def spans_from_bioe(seq: np.ndarray) -> list[tuple[int, int]]:
    """Legal O,B,I,E sequence -> inclusive (start, end) spans."""
    out, start = [], None
    for i, v in enumerate(seq):
        if v == B:
            start = i
        elif v == E:
            if start is not None:
                out.append((start, i))
                start = None
    return out


# ---------------------------------------------------------------------------
# text -> windows -> model -> character spans
# ---------------------------------------------------------------------------

def pack_window_for_inference(input_ids, offsets, w_start, w_end, cls_id, sep_id, pad_id, max_length):
    """Like build_tsawa_dataset.pack_window, minus the training-only `labels`
    column (there is nothing to label at inference time)."""
    content_ids = input_ids[w_start:w_end]
    content_off = offsets[w_start:w_end]
    ids = [cls_id] + content_ids + [sep_id]
    mask = [1] * len(ids)
    pad_n = max_length - len(ids)
    if pad_n:
        ids = ids + [pad_id] * pad_n
        mask = mask + [0] * pad_n
    return ids, mask, content_off


def merge_char_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Union-merge spans that touch or overlap. Spans come from overlapping
    windows, so the same real span can appear (with slightly different edges)
    from two windows; this merges those into one. Known limitation: two
    genuinely distinct spans that happen to touch would also merge. A more
    careful policy (e.g. prefer the copy from whichever window has it most
    centered) is possible but not implemented here."""
    out: list[tuple[int, int]] = []
    for s, e in sorted(spans):
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


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


def run_layer(text: str, tok, model, cfg: LayerConfig, device: str) -> tuple[list[tuple[int, int]], int]:
    """Returns (character spans, number of windows run).

    Two decode policies, per cfg.stitch_first_window:
      - default (Tsawa, Sabche, Chapter, Quotation): decode each window
        independently, then union-merge the resulting character spans. This
        is what each of those models' own published test scores used
        ("each book is decoded once and duplicates from overlapping windows
        are removed").
      - stitched (Yigchung only, as of 2026-09-28): a token's logits come
        only from the first window that covers it (matching how later
        copies were masked out of the training loss), and the whole
        document is Viterbi-decoded once as a single sequence. See
        run_layer_stitched().
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

    all_spans: list[tuple[int, int]] = []
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
            for a, b in tok_spans:
                if a >= n_content or b >= n_content:
                    continue
                cs, ce = content_off[a][0], content_off[b][1]
                if ce > cs:
                    all_spans.append((cs, ce))
    return merge_char_spans(all_spans), len(wins)


def run_layer_stitched(input_ids, offsets, wins, cls_id, sep_id, pad_id, model, cfg: LayerConfig, device) -> list[tuple[int, int]]:
    """Yigchung's own inference recipe: each absolute token index gets its
    logits from the FIRST window (in window order) that covers it -- this
    matches training, where a token repeated in a later, overlapping window
    was masked out of the loss (-100), so the model was never trained to
    produce a meaningful prediction for that token on its second exposure.
    The stitched, whole-document logit sequence is then Viterbi-decoded once,
    rather than decoding each window separately and merging spans afterward."""
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
    out = []
    for a, b in tok_spans:
        cs, ce = offsets[a][0], offsets[b][1]
        if ce > cs:
            out.append((cs, ce))
    return out


# ---------------------------------------------------------------------------
# model loading
# ---------------------------------------------------------------------------

_MODEL_CACHE: dict[str, tuple] = {}


def load_layer_model(name: str, device: str):
    """Returns (tokenizer, model) or raises. Cached across documents in one run."""
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
        # karma689/tibetan-quotation-detection ships a tokenizer_config.json in a
        # newer format (extra_special_tokens as a list) that this transformers
        # version cannot parse (AttributeError on .keys()). Its own README and
        # the Sabche/Tsawa/Chapter cards all state the tokenizer is an unchanged
        # copy of jhu-clsp/mmBERT-base, so falling back to that base copy is not
        # a behavior change, only a workaround for this one repo's broken config.
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
# per-document detection
# ---------------------------------------------------------------------------

@dataclass
class DocResult:
    book_id: str
    text: str
    layers: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    n_windows: dict[str, int] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    elapsed_by_layer: dict[str, float] = field(default_factory=dict)


def detect(text: str, book_id: str, layer_names: list[str], device: str) -> DocResult:
    res = DocResult(book_id=book_id, text=text)
    for name in layer_names:
        t0 = time.time()
        try:
            tok, model = load_layer_model(name, device)
        except Exception as e:  # missing model / load failure: skip and warn
            res.errors[name] = f"model load failed: {e}"
            print(f"  [warn] {book_id}: layer '{name}' skipped: {e}", file=sys.stderr)
            continue
        try:
            spans, n_win = run_layer(text, tok, model, LAYERS[name], device)
            res.layers[name] = spans
            res.n_windows[name] = n_win
        except Exception as e:
            res.errors[name] = f"inference failed: {e}"
            print(f"  [warn] {book_id}: layer '{name}' failed: {e}", file=sys.stderr)
        res.elapsed_by_layer[name] = time.time() - t0
    return res


# ---------------------------------------------------------------------------
# output: JSON
# ---------------------------------------------------------------------------

def write_json(res: DocResult, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{res.book_id}.json"
    payload = {
        "book_id": res.book_id,
        "text_length": len(res.text),
        "layers": {name: [{"start": s, "end": e} for s, e in spans]
                  for name, spans in res.layers.items()},
    }
    if res.errors:
        payload["errors"] = res.errors
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# output: HTML (stacked underlines for overlapping layers)
# ---------------------------------------------------------------------------

def write_html(res: DocResult, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{res.book_id}.html"

    events = []  # (char_pos, delta, layer_name) delta +1 open, -1 close
    for name, spans in res.layers.items():
        for s, e in spans:
            events.append((s, 1, name))
            events.append((e, -1, name))
    bounds = sorted({p for p, _, _ in events} | {0, len(res.text)})

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
        chunk = html_mod.escape(res.text[p:nxt])
        if not chunk:
            continue
        if active:
            shadows = ", ".join(f"inset 0 -{2 + 2 * i}px {LAYERS[n].color}"
                                for i, n in enumerate(sorted(active)))
            title = ", ".join(sorted(active))
            parts.append(f'<span class="span" style="box-shadow:{shadows}" title="{title}">{chunk}</span>')
        else:
            parts.append(chunk)
    # trailing close events at the very end (e == len(text))
    body = "".join(parts)

    counts = {name: len(spans) for name, spans in res.layers.items()}
    legend = "".join(
        f'<div class="legend-row"><span class="swatch" style="background:{LAYERS[n].color}"></span>'
        f'{n} <span class="mut">({counts.get(n, 0)} spans'
        + (f", {res.errors[n]}" if n in res.errors else "")
        + ')</span></div>'
        for n in res.layers.keys() | res.errors.keys()
    )

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>{html_mod.escape(res.book_id)} — layer detection</title>
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
<h1>{html_mod.escape(res.book_id)}</h1>
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
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", type=Path, help="a single text file")
    src.add_argument("--dir", type=Path, help="a folder of .txt files")
    ap.add_argument("--layers", nargs="+", choices=list(LAYERS), default=None)
    ap.add_argument("--all", action="store_true", help="run every layer with a model configured")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--html", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("results"))
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

    files = [args.text] if args.text else sorted(args.dir.glob("*.txt"))
    if not files:
        raise SystemExit(f"no .txt files found in {args.dir}")

    summary = []
    for fp in files:
        text = fp.read_text(encoding="utf-8")
        book_id = fp.stem
        t0 = time.time()
        res = detect(text, book_id, layer_names, args.device)
        elapsed = time.time() - t0

        if args.json:
            p = write_json(res, args.out)
            print(f"wrote {p}")
        if args.html:
            p = write_html(res, args.out)
            print(f"wrote {p}")
        if not args.json and not args.html:
            print(f"\n{book_id}  ({len(text):,} chars, {elapsed:.1f}s)")
            for name in layer_names:
                if name in res.errors:
                    print(f"  {name:10} SKIPPED: {res.errors[name]}")
                else:
                    print(f"  {name:10} {len(res.layers.get(name, [])):4} spans "
                          f"({res.n_windows.get(name, 0)} windows, "
                          f"{res.elapsed_by_layer.get(name, 0):.1f}s)")

        summary.append({
            "book_id": book_id, "text_length": len(text), "elapsed_s": round(elapsed, 2),
            "spans": {n: len(res.layers.get(n, [])) for n in layer_names},
            "errors": res.errors,
        })

    if len(files) > 1:
        print(f"\n{'='*60}\nBATCH SUMMARY — {len(files)} documents")
        for row in summary:
            errs = f"  errors: {row['errors']}" if row["errors"] else ""
            print(f"  {row['book_id']:20} {row['text_length']:8,} chars  {row['elapsed_s']:6.1f}s  "
                  f"{row['spans']}{errs}")
        if args.json:
            (args.out / "_batch_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"wrote {args.out / '_batch_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

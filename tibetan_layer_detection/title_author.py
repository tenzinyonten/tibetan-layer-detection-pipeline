"""Title and author layer: one title and one author per work, found in the
first and last 3,000 characters.

Unlike the other layers this is not a sliding window over the whole book. The
trained models (mmBERT-base) were trained on ONE sequence per work. By default
there are two, each knowing one field with 3 labels (O, B, I):
Yontenn/mmbert-title-only and Yontenn/mmbert-author-only, each decoded on its
own forward pass. The earlier joint model Yontenn/mmbert-title-author-run2
(5 labels: O, B/I-TITLE, B/I-AUTHOR) still works and gives both fields from one
forward pass. The sequence is:

    long work   [CLS] head tokens [SEP] tail tokens [SEP]   (the [SEP] between them is unlabelled)
    short work  [CLS] whole text tokens [SEP]                (a work of 6,000 characters or fewer)

so inference builds exactly that sequence, runs one forward pass, sets the
special tokens aside, decodes with Viterbi (break penalty 6 for a single-entity
model, 12 for the joint one, both picked on validation), keeps one span per field (title = first span,
author = last span) and maps the token spans straight to character offsets in
the full text, so every span is an exact substring of the input.

The crop copies title_author/src/ta_build_gold_dataset.py (regions(), with its
snap_forward / snap_backward). One difference is unavoidable: training also
moved a crop edge that would have cut a gold span; at inference there is no
gold, so the edge stays where the snap puts it.

Title and author come from the trained models only. There is no LLM or paid path.

Nothing here reads a token from a file: Hugging Face access uses the
HF_TOKEN environment variable (or the stored `huggingface-cli login`).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

from .decode import NEG, softmax, span_confidences
from .layer_config import TA_FIELDS, TitleAuthorOptions  # noqa: F401

JOINT_BREAK_PENALTY = 12.0   # the 5-label joint model, picked on validation (the trainer used 5)
SINGLE_BREAK_PENALTY = 6.0   # a single-entity (3-label) model, picked on validation
BASE_TOKENIZER = "jhu-clsp/mmBERT-base"  # the tokenizer the training data was built with
MAX_SEQUENCE = 8192                      # the model's window; head + tail is about 4,100 tokens
SNAP_LIMIT = 600                         # how far a crop edge may move to reach a line start


# ---------------------------------------------------------------------------
# cropping (same rule as the training data)
# ---------------------------------------------------------------------------

def snap_forward(text: str, pos: int, limit: int = SNAP_LIMIT) -> int:
    """First line start at/after pos; else just after the next tsek/shad/space; else pos."""
    i = text.find("\n", pos, pos + limit)
    if i >= 0:
        return i + 1
    for j in range(pos, min(len(text), pos + limit)):
        if text[j] in "་། ":
            return j + 1
    return pos


def snap_backward(text: str, pos: int, limit: int = SNAP_LIMIT) -> int:
    """Last line start at/before pos; else just after the previous tsek/shad/space; else pos."""
    i = text.rfind("\n", max(0, pos - limit), pos + 1)
    if i >= 0:
        return i + 1
    for j in range(pos, max(0, pos - limit), -1):
        if text[j - 1] in "་། ":
            return j
    return pos


def crop_regions(text: str, k: int = 3000) -> list[tuple[str, int, int]]:
    """[("whole", 0, L)] for a work of 2k characters or fewer, or if head and
    tail would meet; otherwise [("head", 0, e), ("tail", st, L)]."""
    n = len(text)
    if n <= 2 * k:
        return [("whole", 0, n)]
    e = snap_forward(text, k)
    st = snap_backward(text, n - k)
    return [("whole", 0, n)] if e >= st else [("head", 0, e), ("tail", st, n)]


# ---------------------------------------------------------------------------
# sequence building and decoding
# ---------------------------------------------------------------------------

def special_ids(tok) -> tuple[int, int]:
    cls_id = tok.cls_token_id if tok.cls_token_id is not None else tok.bos_token_id
    sep_id = tok.sep_token_id if tok.sep_token_id is not None else tok.eos_token_id
    return cls_id, sep_id


def build_sequence(text: str, tok, k: int = 3000, strip_underscore: bool = False) -> dict:
    """The one training-format sequence for this work.
    strip_underscore: the single-entity models were trained on text with the
    "_" characters removed (a syllable-separator artifact of some sources;
    ta_strip_underscore.py), so they are removed from each cropped region before
    tokenizing. Token offsets are still returned in the ORIGINAL text, so a span
    is an exact substring of the input (it may contain a "_" inside).
    Returns {"input_ids": [...], "real": positions of real (non-special)
    tokens, "offsets": (start, end) in `text` for each real token,
    "regions": the crop}."""
    cls_id, sep_id = special_ids(tok)
    regions = crop_regions(text, k)
    ids, real, offsets = [cls_id], [], []
    for n, (_, s, e) in enumerate(regions):
        chunk = text[s:e]
        keep = [i for i, c in enumerate(chunk) if c != "_"] if strip_underscore and "_" in chunk else None
        enc = tok("".join(chunk[i] for i in keep) if keep is not None else chunk,
                  add_special_tokens=False, return_offsets_mapping=True)
        for tid, (a, b) in zip(enc["input_ids"], enc["offset_mapping"]):
            real.append(len(ids))
            ids.append(tid)
            if keep is None:
                offsets.append((s + a, s + b))
            else:  # map the token's span in the stripped text back to the original characters
                oa = s + keep[min(a, len(keep) - 1)]
                offsets.append((oa, s + keep[b - 1] + 1 if b > a else oa))
        ids.append(sep_id)  # after the head, and after the last region
    if len(ids) > MAX_SEQUENCE:
        raise ValueError(f"title/author input is {len(ids)} tokens, over the model's {MAX_SEQUENCE}")
    return {"input_ids": ids, "real": real, "offsets": offsets, "regions": regions}


def _label_parts(id2label: dict[int, str]) -> list[tuple[str, str | None]]:
    """Per label id: ("O", None), ("B", entity) or ("I", entity). A bare "B" / "I"
    (a single-entity model that does not name its entity) has entity None."""
    parts = []
    for i in range(len(id2label)):
        name = str(id2label[i]).upper()
        if name == "O":
            parts.append(("O", None))
        elif name[:2] in ("B-", "I-"):
            parts.append((name[0], name[2:]))
        elif name in ("B", "I"):
            parts.append((name, None))
        else:
            raise ValueError(f"unrecognised label {id2label[i]!r}")
    return parts


def transition_matrix(id2label: dict[int, str], break_penalty: float) -> np.ndarray:
    """m[i, j] = cost of moving from label i to label j. I-X may only follow B-X
    or I-X; every way of leaving a span (B/I -> anything but I of the same
    entity) costs `break_penalty`. For the 5-label model this is exactly the
    matrix in title_author/src/ta_mmbert_score.py."""
    parts = _label_parts(id2label)
    n = len(parts)
    m = np.zeros((n, n))
    for i, (pi, ei) in enumerate(parts):
        for j, (pj, ej) in enumerate(parts):
            if pj == "I" and not (pi in ("B", "I") and ei == ej):
                m[i, j] = NEG
            elif pi in ("B", "I") and not (pj == "I" and ej == ei):
                m[i, j] -= break_penalty
    return m


def viterbi(logits: np.ndarray, id2label: dict[int, str], break_penalty: float) -> np.ndarray:
    parts = _label_parts(id2label)
    trans = transition_matrix(id2label, break_penalty)
    T, C = logits.shape
    dp = np.full((T, C), NEG)
    bp = np.zeros((T, C), dtype=np.int64)
    dp[0] = logits[0]
    for c, (p, _) in enumerate(parts):
        if p == "I":
            dp[0, c] = NEG  # a sequence cannot open mid-span
    for t in range(1, T):
        s = dp[t - 1][:, None] + trans
        bp[t] = s.argmax(0)
        dp[t] = s.max(0) + logits[t]
    path = np.zeros(T, dtype=np.int64)
    path[-1] = int(dp[-1].argmax())
    for t in range(T - 1, 0, -1):
        path[t - 1] = bp[t, path[t]]
    return path


def entity_spans(path: np.ndarray, id2label: dict[int, str], entity: str | None) -> list[tuple[int, int]]:
    """Inclusive token spans of one entity (None = the unnamed entity)."""
    parts = _label_parts(id2label)
    out, start = [], None
    for k, lab in enumerate(path):
        p, e = parts[int(lab)]
        if p == "B" and e == entity:
            if start is not None:
                out.append((start, k - 1))
            start = k
        elif p == "I" and e == entity:
            if start is None:
                start = k
        else:
            if start is not None:
                out.append((start, k - 1))
                start = None
    if start is not None:
        out.append((start, len(path) - 1))
    return out


def entity_for(field: str, id2label: dict[int, str]) -> tuple[bool, str | None]:
    """(usable, entity) -- which entity of this model answers `field`. A joint
    model names both; a single-entity model answers its own field (named or not)."""
    ents = {e for p, e in _label_parts(id2label) if p != "O"}
    if field.upper() in ents:
        return True, field.upper()
    if ents == {None}:
        return True, None
    return False, None


def default_break_penalty(id2label: dict[int, str]) -> float:
    """6 for a single-entity model (one field), 12 for the joint model (both)."""
    entities = {e for p, e in _label_parts(id2label) if p != "O"}
    return JOINT_BREAK_PENALTY if len(entities) > 1 else SINGLE_BREAK_PENALTY


def decode_fields(text: str, seq: dict, logits: np.ndarray, id2label: dict[int, str],
                  fields: list[str], break_penalty: float | None = None,
                  author_span: str = "last") -> dict[str, list[dict]]:
    """logits: [len(input_ids), n_labels] for the whole sequence. Sets the special
    tokens aside, decodes, and returns {field: [span]} with at most one span per
    field (title = first span, author = last), as character offsets into `text`."""
    real = np.array(seq["real"])
    if len(real) == 0:
        return {f: [] for f in fields}
    lg = logits[real].astype(np.float64)
    path = viterbi(lg, id2label, default_break_penalty(id2label) if break_penalty is None else break_penalty)
    probs = softmax(lg)
    out: dict[str, list[dict]] = {}
    head_end = seq["regions"][0][2] if len(seq["regions"]) == 2 else None
    for f in fields:
        ok, ent = entity_for(f, id2label)
        spans = entity_spans(path, id2label, ent) if ok else []
        if not spans:
            out[f] = []
            continue
        a, b = spans[-1] if (f == "author" and author_span == "last") else spans[0]
        s, e = seq["offsets"][a][0], seq["offsets"][b][1]
        while s < e and text[s].isspace():  # a token carries the space before its word
            s += 1
        conf = span_confidences(probs, path, [(a, b)])[0]
        block = "whole" if head_end is None else ("head" if s < head_end else "tail")
        out[f] = [{"start": s, "end": e, "confidence": conf, "window_votes": 1, "block": block}]
    return out


# ---------------------------------------------------------------------------
# the trained model
# ---------------------------------------------------------------------------

_MODEL_CACHE: dict[str, tuple] = {}


def load_ta_model(repo: str, device: str):
    """(tokenizer, model) for a title/author model: a local folder, or a Hugging
    Face repo. Weights live in the repo's best/ subfolder (pointing at the repo
    root fails with "Unrecognized model... model_type"), so that is what loads.
    Access uses HF_TOKEN from the environment (or the stored login)."""
    if repo in _MODEL_CACHE:
        return _MODEL_CACHE[repo]
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    token = os.environ.get("HF_TOKEN")
    if Path(repo).is_dir():
        root = Path(repo)
    else:
        from huggingface_hub import snapshot_download
        root = Path(snapshot_download(repo, token=token, allow_patterns=["best/*"]))
        if not (root / "best" / "config.json").is_file():  # a model saved at the repo root
            root = Path(snapshot_download(repo, token=token, allow_patterns=[
                "config.json", "*.safetensors", "*.bin"]))
    mdir = root / "best" if (root / "best" / "config.json").is_file() else root
    tok = AutoTokenizer.from_pretrained(BASE_TOKENIZER, token=token)
    if not tok.is_fast:
        raise RuntimeError(f"{BASE_TOKENIZER}: tokenizer is not a fast tokenizer (need offset mapping)")
    tok.model_max_length = int(1e12)
    model = AutoModelForTokenClassification.from_pretrained(mdir).eval().to(device)
    _MODEL_CACHE[repo] = (tok, model)
    return tok, model


def uses_stripped_text(id2label: dict[int, str]) -> bool:
    """A single-entity model was trained on underscore-free text, the joint run-2
    model on the text as it was."""
    return len({e for p, e in _label_parts(id2label) if p != "O"}) == 1


def run_model(text: str, fields: list[str], repo: str, device: str, k: int = 3000,
              break_penalty: float | None = None,
              strip_underscore: bool | None = None, author_span: str = "last") -> dict[str, list[dict]]:
    import torch

    tok, model = load_ta_model(repo, device)
    id2label = {int(i): n for i, n in model.config.id2label.items()}
    strip = uses_stripped_text(id2label) if strip_underscore is None else strip_underscore
    seq = build_sequence(text, tok, k, strip)
    with torch.no_grad():
        x = torch.tensor([seq["input_ids"]], device=device)
        logits = model(input_ids=x).logits[0].float().cpu().numpy()
    out = decode_fields(text, seq, logits, id2label, fields, break_penalty, author_span)
    for f in fields:
        for sp in out[f]:
            sp["model"] = repo
    return out


# ---------------------------------------------------------------------------
# all requested fields
# ---------------------------------------------------------------------------

def run_title_author(text: str, fields: list[str], opts: TitleAuthorOptions, device: str
                     ) -> tuple[dict[str, list[dict]], dict[str, float], dict[str, str]]:
    """Returns (layers, elapsed_s, errors). `fields` is a subset of ("title", "author").
    The trained model of each field runs once per distinct model (a joint model that
    serves both fields needs one forward pass). A failure is an error for that field."""
    layers: dict[str, list[dict]] = {}
    elapsed: dict[str, float] = {}
    errors: dict[str, str] = {}
    model_of = {"title": opts.title_model, "author": opts.author_model}
    for repo in dict.fromkeys(model_of[f] for f in fields):  # one forward pass per distinct model
        mine = [f for f in fields if model_of[f] == repo]
        t0 = time.time()
        try:
            got = run_model(text, mine, repo, device, opts.crop_chars, opts.break_penalty, opts.strip_underscore,
                            author_span=opts.author_span)
        except Exception as e:
            for f in mine:
                errors[f] = f"model load or inference failed: {e}"
            print(f"  [warn] title/author model {repo} failed: {e}", file=sys.stderr)
            continue
        for f in mine:
            layers[f] = got[f]
            elapsed[f] = round(time.time() - t0, 2)
    return layers, elapsed, errors

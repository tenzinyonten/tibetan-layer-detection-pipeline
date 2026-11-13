"""Title and author layer: one title and one author per work, found in the
first and last 3,000 characters.

Unlike the other layers this is not a sliding window over the whole book. The
trained model (Yontenn/mmbert-title-author-run2, mmBERT-base, 5 labels O,
B/I-TITLE, B/I-AUTHOR) was trained on ONE sequence per work:

    long work   [CLS] head tokens [SEP] tail tokens [SEP]   (the [SEP] between them is unlabelled)
    short work  [CLS] whole text tokens [SEP]                (a work of 6,000 characters or fewer)

so inference builds exactly that sequence, runs one forward pass, sets the
special tokens aside, decodes with Viterbi (break penalty 12, picked on the
trainer's validation split), keeps one span per field (title = first span,
author = last span) and maps the token spans straight to character offsets in
the full text, so every span is an exact substring of the input.

The crop copies title_author/src/ta_build_gold_dataset.py (regions(), with its
snap_forward / snap_backward). One difference is unavoidable: training also
moved a crop edge that would have cut a gold span; at inference there is no
gold, so the edge stays where the snap puts it.

A second system, the LLM skill in title_author/skills/tibetan-title-author/, can
be used instead of, or next to, the model (TitleAuthorOptions.system). It is
imported only then, and it costs a paid LLM call per work.

Nothing here reads a token or key from a file: Hugging Face access uses the
HF_TOKEN environment variable (or the stored `huggingface-cli login`), and the
skill's keys are GEMINI_API_KEY / ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

from .decode import NEG, softmax, span_confidences
from .layer_config import TA_FIELDS, TitleAuthorOptions

BASE_TOKENIZER = "jhu-clsp/mmBERT-base"  # the tokenizer the training data was built with
MAX_SEQUENCE = 8192                      # the model's window; head + tail is about 4,100 tokens
SNAP_LIMIT = 600                         # how far a crop edge may move to reach a line start

SKILL_SUBDIR = "title_author/skills/tibetan-title-author/scripts"


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


def build_sequence(text: str, tok, k: int = 3000) -> dict:
    """The one training-format sequence for this work.
    Returns {"input_ids": [...], "real": positions of real (non-special)
    tokens, "offsets": (start, end) in `text` for each real token,
    "regions": the crop}."""
    cls_id, sep_id = special_ids(tok)
    regions = crop_regions(text, k)
    ids, real, offsets = [cls_id], [], []
    for n, (_, s, e) in enumerate(regions):
        enc = tok(text[s:e], add_special_tokens=False, return_offsets_mapping=True)
        for tid, (a, b) in zip(enc["input_ids"], enc["offset_mapping"]):
            real.append(len(ids))
            ids.append(tid)
            offsets.append((s + a, s + b))
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


def decode_fields(text: str, seq: dict, logits: np.ndarray, id2label: dict[int, str],
                  fields: list[str], break_penalty: float = 12.0) -> dict[str, list[dict]]:
    """logits: [len(input_ids), n_labels] for the whole sequence. Sets the special
    tokens aside, decodes, and returns {field: [span]} with at most one span per
    field (title = first span, author = last), as character offsets into `text`."""
    real = np.array(seq["real"])
    if len(real) == 0:
        return {f: [] for f in fields}
    lg = logits[real].astype(np.float64)
    path = viterbi(lg, id2label, break_penalty)
    probs = softmax(lg)
    out: dict[str, list[dict]] = {}
    head_end = seq["regions"][0][2] if len(seq["regions"]) == 2 else None
    for f in fields:
        ok, ent = entity_for(f, id2label)
        spans = entity_spans(path, id2label, ent) if ok else []
        if not spans:
            out[f] = []
            continue
        a, b = spans[-1] if f == "author" else spans[0]
        s, e = seq["offsets"][a][0], seq["offsets"][b][1]
        while s < e and text[s].isspace():  # a token carries the space before its word
            s += 1
        conf = span_confidences(probs, path, [(a, b)])[0]
        block = "whole" if head_end is None else ("head" if s < head_end else "tail")
        out[f] = [{"start": s, "end": e, "confidence": conf, "window_votes": 1, "block": block,
                   "system": "model"}]
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


def run_model(text: str, fields: list[str], repo: str, device: str, k: int = 3000,
              break_penalty: float = 12.0) -> dict[str, list[dict]]:
    import torch

    tok, model = load_ta_model(repo, device)
    seq = build_sequence(text, tok, k)
    with torch.no_grad():
        x = torch.tensor([seq["input_ids"]], device=device)
        logits = model(input_ids=x).logits[0].float().cpu().numpy()
    id2label = {int(i): n for i, n in model.config.id2label.items()}
    out = decode_fields(text, seq, logits, id2label, fields, break_penalty)
    for f in fields:
        for sp in out[f]:
            sp["model"] = repo
    return out


# ---------------------------------------------------------------------------
# the skill (optional, paid)
# ---------------------------------------------------------------------------

def _skill_modules(skill_dir: str | None):
    """Import the skill lazily; it is never loaded in model-only runs."""
    d = Path(skill_dir or os.environ.get("TA_SKILL_DIR") or Path(__file__).resolve().parents[1] / SKILL_SUBDIR)
    if not (d / "title_author_skill.py").is_file():
        raise RuntimeError(f"title/author skill not found in {d}; set TA_SKILL_DIR or --title-author-skill-dir")
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
    import title_author_skill as S  # noqa: E402
    return S


def run_skill(text: str, fields: list[str], opts: TitleAuthorOptions, call_llm=None,
              k: int = 3000) -> dict[str, list[dict]]:
    """Ask the skill for the title and author of this work and locate its answers
    in `text`. The skill receives the same crop as the model. It returns strings
    plus a region; the title is found at its first occurrence in that block and
    the author at its last. Spans carry review_needed / review_reasons from the
    skill's own flags and no confidence."""
    S = _skill_modules(opts.skill_dir)
    regions = crop_regions(text, k)
    if len(regions) == 1:
        head, tail, whole = text, "", True
    else:
        head, tail, whole = text[regions[0][1]:regions[0][2]], text[regions[1][1]:regions[1][2]], False
    nh, nt, nw = S.normalize_input(head, tail, whole)  # the skill trims each block to 3,000 characters
    tail_base = (regions[1][1] + len(tail) - len(nt)) if tail else (len(head) - len(nt))
    call = call_llm if call_llm is not None else S.llm_for(opts.skill_model)
    ans = S.extract(head, tail, whole, call_llm=call)
    out: dict[str, list[dict]] = {f: [] for f in fields}
    for f in fields:
        val, region = ans.get(f), ans.get(f + "_region")
        if not val:
            continue
        block, base = (nh, 0) if (region == "head" or nw) else (nt, tail_base)
        i = block.rfind(val) if f == "author" else block.find(val)
        if i < 0:
            continue
        out[f] = [{"start": base + i, "end": base + i + len(val), "confidence": None, "window_votes": 1,
                   "block": "whole" if len(regions) == 1 else ("head" if base == 0 else "tail"),
                   "system": "skill", "review_needed": bool(ans.get("needs_review")),
                   "review_reasons": list(ans.get("review_reasons") or [])}]
    return out


# ---------------------------------------------------------------------------
# the two systems together
# ---------------------------------------------------------------------------

def _iou(a: dict, b: dict) -> float:
    inter = max(0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
    union = (a["end"] - a["start"]) + (b["end"] - b["start"]) - inter
    return inter / union if union else 0.0


def run_title_author(text: str, fields: list[str], opts: TitleAuthorOptions, device: str,
                     call_llm=None) -> tuple[dict[str, list[dict]], dict[str, float], dict[str, str]]:
    """Returns (layers, elapsed_s, errors). `fields` is a subset of ("title", "author").
    system "model": layers[field] from the trained model.
    system "skill": layers[field] from the skill.
    system "both":  layers[field] from the model and layers[field + "_skill"] from
                    the skill, side by side; each span gets systems_agree
                    (IoU >= 0.5 between the two, the benchmark's own rule)."""
    layers: dict[str, list[dict]] = {}
    elapsed: dict[str, float] = {}
    errors: dict[str, str] = {}
    model_of = {"title": opts.title_model, "author": opts.author_model}
    result: dict[str, dict[str, list[dict]]] = {}
    if opts.system in ("model", "both"):
        for repo in dict.fromkeys(model_of[f] for f in fields):  # one forward pass per distinct model
            mine = [f for f in fields if model_of[f] == repo]
            t0 = time.time()
            try:
                got = run_model(text, mine, repo, device, opts.crop_chars, opts.break_penalty)
            except Exception as e:
                for f in mine:
                    errors[f] = f"model load or inference failed: {e}"
                print(f"  [warn] title/author model {repo} failed: {e}", file=sys.stderr)
                continue
            for f in mine:
                layers[f] = got[f]
                elapsed[f] = round(time.time() - t0, 2)
        result["model"] = {f: layers.get(f, []) for f in fields}
    if opts.system in ("skill", "both"):
        t0 = time.time()
        try:
            got = run_skill(text, fields, opts, call_llm, opts.crop_chars)
        except Exception as e:
            for f in fields:
                errors[f if opts.system == "skill" else f + "_skill"] = f"skill failed: {e}"
            print(f"  [warn] title/author skill failed: {e}", file=sys.stderr)
        else:
            for f in fields:
                layers[f if opts.system == "skill" else f + "_skill"] = got[f]
                elapsed[f if opts.system == "skill" else f + "_skill"] = round(time.time() - t0, 2)
            result["skill"] = got
    if opts.system == "both" and "model" in result and "skill" in result:
        for f in fields:
            m, s = result["model"].get(f, []), result["skill"].get(f, [])
            agree = bool(m and s and _iou(m[0], s[0]) >= 0.5)
            for sp in m + s:
                sp["systems_agree"] = agree
    return layers, elapsed, errors

"""BIO and BIOE Viterbi decoding, plus span confidence.

transition_matrix / viterbi / spans_from_bio are extracted, unchanged, from
tenzinyonten/layer_detection's common/eval_viterbi_iou.py, so this repo does
not need that file's heavier dependency (the `datasets` library) just for
three small numpy functions. bioe_transition_matrix / viterbi_bioe /
spans_from_bioe (Quotation's 4-state scheme) and softmax / span_confidences
are this project's own code (no BIOE decoder existed anywhere before this
was first written inline in detect_layers.py); merge_char_spans is the
union-merge of overlapping-window spans used by infer.py. The Yigchung
stitched-window decode itself lives in infer.py, since it also needs
per-window model calls, not just the decode step."""

from __future__ import annotations

import numpy as np

NEG = -1.0e9
O, B, I = 0, 1, 2


def transition_matrix(break_penalty: float) -> np.ndarray:
    """
    3-label BIO transitions. I may only follow B or I; every span exit costs
    `break_penalty`, so fragmenting a span is penalised.
    """
    labels = ["O", "B", "I"]
    n = len(labels)
    m = np.zeros((n, n), dtype=np.float64)
    for i, prev in enumerate(labels):
        for j, nxt in enumerate(labels):
            if nxt == "I" and prev not in ("B", "I"):
                m[i, j] = NEG
                continue
            if prev == "O":
                continue
            # leaving a span (B/I -> anything that is not I) costs the penalty
            if nxt != "I":
                m[i, j] -= break_penalty
    return m


def viterbi(logits: np.ndarray, break_penalty: float) -> np.ndarray:
    """logits [T, C] -> best legal label sequence [T]."""
    T, C = logits.shape
    trans = transition_matrix(break_penalty)
    dp = np.full((T, C), NEG)
    bp = np.zeros((T, C), dtype=np.int64)
    dp[0] = logits[0]
    dp[0, I] = NEG  # a window cannot open mid-span
    for t in range(1, T):
        scores = dp[t - 1][:, None] + trans  # [C_prev, C_next]
        bp[t] = scores.argmax(axis=0)
        dp[t] = scores.max(axis=0) + logits[t]
    path = np.zeros(T, dtype=np.int64)
    path[-1] = int(dp[-1].argmax())
    for t in range(T - 1, 0, -1):
        path[t - 1] = bp[t, path[t]]
    return path


def spans_from_bio(seq: np.ndarray) -> list[tuple[int, int]]:
    """BIO -> inclusive (start, end) spans, matching the team's convention."""
    out, start = [], None
    for i, v in enumerate(seq):
        if v == B:
            if start is not None:
                out.append((start, i - 1))
            start = i
        elif v == I:
            if start is None:
                start = i
        else:
            if start is not None:
                out.append((start, i - 1))
                start = None
    if start is not None:
        out.append((start, len(seq) - 1))
    return out


# ---------------------------------------------------------------------------
# BIOE Viterbi (Quotation only) and confidence, added when detect_layers.py
# was split into three pipeline stages. Label ids match
# BDRC/Bo-Quotation-Detection's config.json and train_layer.py's "bioe"
# scheme (ENTITIES["bioe"] = (b=1, i=2, e=3)): O=0 B=1 I=2 E=3. Legal
# grammar: a span is exactly B I* E (an I never opens or closes a span, and
# O never continues one). First token may not be I or E; a token after E
# may be O (span closed) or a fresh B (new span starts immediately).
# ---------------------------------------------------------------------------

_E = 3  # BIOE's fourth state; O/B/I ids above are shared with the BIO decoder


def bioe_transition_matrix(break_penalty: float) -> np.ndarray:
    """m[i, j] = cost of moving FROM state i TO state j (added to j's score)."""
    m = np.zeros((4, 4))
    legal = {
        (O, O), (O, B),
        (B, I), (B, _E),
        (I, I), (I, _E),
        (_E, O), (_E, B),
    }
    for i in range(4):
        for j in range(4):
            if (i, j) not in legal:
                m[i, j] = NEG
    m[_E, O] -= break_penalty  # the only legal way to end a span
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
    dp[0, _E] = NEG
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
        elif v == _E:
            if start is not None:
                out.append((start, i))
                start = None
    return out


def softmax(logits: np.ndarray) -> np.ndarray:
    m = logits.max(axis=-1, keepdims=True)
    e = np.exp(logits - m)
    return e / e.sum(axis=-1, keepdims=True)


def span_confidences(probs: np.ndarray, seq: np.ndarray,
                     tok_spans: list[tuple[int, int]]) -> list[float]:
    """Mean softmax probability of the DECODED (Viterbi-chosen) label over
    each span's tokens. Not argmax confidence: Viterbi can legally pick a
    lower-scoring label to satisfy the BIO/BIOE grammar (e.g. continuing a
    span through a token whose argmax was O), and it's that chosen label's
    probability the span should be judged on, not whatever argmax preferred."""
    out = []
    for a, b in tok_spans:
        p = probs[a:b + 1]
        chosen = seq[a:b + 1]
        out.append(float(p[np.arange(len(chosen)), chosen].mean()))
    return out


def merge_char_spans(spans: list[tuple[int, int, float]]) -> list[tuple[int, int, float]]:
    """Union-merge spans that touch or overlap. Spans come from overlapping
    windows, so the same real span can appear (with slightly different edges)
    from two windows; this merges those into one, combining confidence as a
    character-length-weighted mean of the merged pieces (an approximation:
    when three or more pieces chain together the result depends on merge
    order, since each new piece folds into the running accumulated span
    rather than all pieces being averaged at once). Known limitation: two
    genuinely distinct spans that happen to touch would also merge."""
    out: list[tuple[int, int, float]] = []
    for s, e, c in sorted(spans, key=lambda x: (x[0], x[1])):
        if out and s <= out[-1][1]:
            os_, oe, oc = out[-1]
            ol, nl = oe - os_, e - s
            merged_c = (oc * ol + c * nl) / (ol + nl) if (ol + nl) else oc
            out[-1] = (os_, max(oe, e), merged_c)
        else:
            out.append((s, e, c))
    return out

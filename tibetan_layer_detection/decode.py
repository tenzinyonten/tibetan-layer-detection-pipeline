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


def overlap_regions(window_char_ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Given each window's (start_char, end_char) content range, returns the
    char ranges where two consecutive windows overlap. Windows are sorted by
    start first since the sliding-window list's last ("flush-right") entry
    can start earlier than the one before it."""
    ranges = sorted(window_char_ranges)
    regions = []
    for i in range(len(ranges) - 1):
        if ranges[i + 1][0] < ranges[i][1]:
            regions.append((ranges[i + 1][0], ranges[i][1]))
    return regions


def merge_spans_with_votes(
    spans_with_window: list[tuple[int, int, float, int]],
    regions: list[tuple[int, int]],
) -> list[tuple[int, int, float, int]]:
    """Union-merge spans (start, end, confidence, window_index) that touch or
    overlap, same rule as merge_char_spans, while also counting how many
    distinct windows contributed to each merged span ("window_votes") and
    adjusting confidence by that count:
      - 2+ windows agree on (overlapping pieces of) the span -> confidence
        boosted by 0.1, capped at 1.0.
      - exactly 1 window produced it AND the span falls inside a region
        another window also covered (so that window saw the same text but
        did not predict a span there) -> confidence reduced by 0.05.
      - exactly 1 window produced it and it is NOT inside any overlap
        region (e.g. a document edge, or the whole document fit in one
        window) -> confidence unchanged; there was no second window to
        agree or disagree.
    Returns (start, end, adjusted_confidence, window_votes)."""
    merged: list[list] = []
    for s, e, c, w in sorted(spans_with_window, key=lambda x: (x[0], x[1])):
        if merged and s <= merged[-1][1]:
            os_, oe, oc, ow = merged[-1]
            ol, nl = oe - os_, e - s
            merged_c = (oc * ol + c * nl) / (ol + nl) if (ol + nl) else oc
            ow.add(w)
            merged[-1] = [os_, max(oe, e), merged_c, ow]
        else:
            merged.append([s, e, c, {w}])

    out = []
    for s, e, c, ow in merged:
        votes = len(ow)
        if votes >= 2:
            c = min(1.0, c + 0.1)
        elif any(s < r_end and e > r_start for r_start, r_end in regions):
            c = max(0.0, c - 0.05)
        out.append((s, e, c, votes))
    return out


# ---------------------------------------------------------------------------
# post-Viterbi linguistic cleanup rules (tsheg/shad-aware), applied to the
# character spans a layer produced, before those spans are written out.
# ---------------------------------------------------------------------------

TSHEG = "་"
SHAD = "།"
_GAP_PUNCT = {TSHEG, SHAD}
TSHEG_LOOKAHEAD = 20   # how far past a truncated span's end to look for the
                       # next tsheg before giving up and leaving it alone
MAX_MERGE_GAP = 10     # merge two spans only if the punctuation/whitespace
                       # gap between them is shorter than this many characters
MIN_SPAN_CHARS = 5     # spans shorter than this are dropped as noise


def _is_gap_ok(gap: str) -> bool:
    return len(gap) > 0 and all(ch in _GAP_PUNCT or ch.isspace() for ch in gap)


def extend_to_next_tsheg(
    spans: list[tuple[int, int, float, int]], text: str
) -> list[tuple[int, int, float, int]]:
    """A span that doesn't end right after a tsheg or shad looks like it cuts
    a syllable in half; extend its end to just past the next tsheg, as long
    as that tsheg is within TSHEG_LOOKAHEAD characters. A span already ending
    cleanly (right after a tsheg/shad, or at the end of the text) is left as
    is, as is one with no tsheg within the lookahead window."""
    out = []
    for s, e, c, v in spans:
        if e >= len(text) or text[e - 1] in (TSHEG, SHAD):
            out.append((s, e, c, v))
            continue
        new_e = e
        for j in range(e, min(e + TSHEG_LOOKAHEAD, len(text))):
            if text[j] == TSHEG:
                new_e = j + 1
                break
        out.append((s, new_e, c, v))
    return out


def merge_small_gaps(
    spans: list[tuple[int, int, float, int]], text: str
) -> list[tuple[int, int, float, int]]:
    """Merge two spans in the same layer if the text between them is under
    MAX_MERGE_GAP characters and made up only of tsheg, shad, or whitespace --
    i.e. they are almost certainly one span the model only split because of a
    punctuation mark. Also merges spans that already touch or overlap (gap
    length 0): extend_to_next_tsheg can push two nearby spans' ends out to
    the same tsheg, which otherwise leaves overlapping spans in the output
    (an empty gap has no characters to satisfy the punctuation-only check,
    so it needs this separate always-merge case). Confidence of the merged
    span is a character-length-weighted mean of the two pieces (same
    convention as merge_char_spans); window_votes takes the stronger of the
    two spans' vote counts."""
    out: list[tuple[int, int, float, int]] = []
    for s, e, c, v in sorted(spans, key=lambda x: (x[0], x[1])):
        if out:
            ps, pe, pc, pv = out[-1]
            touching_or_overlapping = s <= pe
            gap_ok = touching_or_overlapping or (
                len(text[pe:s]) < MAX_MERGE_GAP and _is_gap_ok(text[pe:s]))
            if gap_ok:
                pl, nl = pe - ps, e - s
                merged_c = (pc * pl + c * nl) / (pl + nl) if (pl + nl) else pc
                out[-1] = (ps, max(pe, e), merged_c, max(pv, v))
                continue
        out.append((s, e, c, v))
    return out


def remove_short_spans(
    spans: list[tuple[int, int, float, int]], min_len: int = MIN_SPAN_CHARS
) -> tuple[list[tuple[int, int, float, int]], int]:
    """Drops spans under min_len characters as likely noise. Returns
    (kept_spans, n_removed)."""
    kept = [sp for sp in spans if sp[1] - sp[0] >= min_len]
    return kept, len(spans) - len(kept)


FRAGMENT_LOOKAHEAD = 250  # how far repair_fragments looks for the end of the heading line


def repair_fragments(
    spans: list[tuple[int, int, float, int]], text: str, extend: bool = False,
    min_len: int = MIN_SPAN_CHARS, lookahead: int = FRAGMENT_LOOKAHEAD,
) -> tuple[list[tuple[int, int, float, int]], int]:
    """For a span shorter than min_len (a fragment: the model opened a heading
    and stopped after its first token or two):
      1. if a longer span in the same list contains it, it is a duplicate of
         that span and is dropped;
      2. otherwise, only when extend=True, extend its end to the next shad or
         newline (a heading is a line up to its closing shad; the shad itself
         stays outside the span, as in the gold). Left unchanged if neither is
         within `lookahead`. Identical results from two fragments are kept once.
    Without extend, fragments with no cover are left for remove_short_spans.
    Returns (spans, n_dropped_as_duplicates)."""
    longer = [(s, e) for s, e, _, _ in spans if e - s >= min_len]
    out, n_dup, seen = [], 0, set()
    for s, e, c, v in spans:
        if e - s >= min_len:
            out.append((s, e, c, v))
            continue
        if any(ls <= s and e <= le for ls, le in longer):
            n_dup += 1
            continue
        if extend:
            j, lim = e, min(len(text), e + lookahead)
            while j < lim and text[j] not in (SHAD, "\n"):
                j += 1
            if j < len(text) and text[j] in (SHAD, "\n") and (s, j) not in seen:
                seen.add((s, j))
                out.append((s, j, c, v))
                continue
        out.append((s, e, c, v))
    return out, n_dup


def apply_linguistic_rules(
    spans: list[tuple[int, int, float, int]], text: str, *,
    extend_to_tsheg: bool = True, merge_gaps: bool = True, repair: bool = False,
    extend_fragments: bool = False,
) -> tuple[list[tuple[int, int, float, int]], int]:
    """Post-Viterbi cleanup, in order: extend truncated spans to the next
    tsheg (extend_to_tsheg), merge spans separated only by a short run of
    punctuation/whitespace (merge_gaps), repair fragments (repair: see
    repair_fragments), then drop any span still under MIN_SPAN_CHARS
    characters as noise. The three switches come from the layer's
    LayerConfig (extend_to_tsheg / merge_small_gaps / repair_fragments); the
    defaults here are the original behavior. Returns (cleaned_spans,
    n_removed), counting both fragments dropped as duplicates and short spans
    dropped as noise."""
    n_dup = 0
    if extend_to_tsheg:
        spans = extend_to_next_tsheg(spans, text)
    if merge_gaps:
        spans = merge_small_gaps(spans, text)
    if repair:
        spans, n_dup = repair_fragments(spans, text, extend=extend_fragments)
    spans, n_removed = remove_short_spans(spans)
    return spans, n_removed + n_dup


# ---------------------------------------------------------------------------
# --dual-window: reconciling a full-size run against a half-size run
# ---------------------------------------------------------------------------

def _iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    s, e = max(a[0], b[0]), min(a[1], b[1])
    inter = max(0, e - s)
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union else 0.0


def merge_dual_window(
    full_spans: list[tuple[int, int, float, int]],
    half_spans: list[tuple[int, int, float, int]],
    iou_threshold: float = 0.5,
) -> list[tuple[int, int, float, int, str]]:
    """Reconciles the full-window run's spans against the half-window run's
    (each already through apply_linguistic_rules). Two spans, one from each
    run, are the "same" span if their character-overlap IoU is at least
    iou_threshold (0.5 by default: more than half of their combined extent
    overlaps) -- window-size changes shift span edges by a few characters,
    so exact-position matching would be too strict.

    Returns (start, end, confidence, window_votes, dual_window_agreement)
    5-tuples, agreement being:
      "both"      - matched in both runs: high confidence. Kept at the
                    full-window run's edges (the configured/primary window
                    size), confidence averaged across both runs, window_votes
                    the stronger of the two.
      "full_only" - only the full-window run produced it.
      "half_only" - only the half-window run produced it.
    full_only/half_only spans are exactly the ones the caller should flag for
    review: the two window sizes disagree on them."""
    used_half: set[int] = set()
    out: list[tuple[int, int, float, int, str]] = []
    for fs in full_spans:
        best_j, best_iou = None, 0.0
        for j, hs in enumerate(half_spans):
            if j in used_half:
                continue
            iou = _iou((fs[0], fs[1]), (hs[0], hs[1]))
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_j is not None and best_iou >= iou_threshold:
            hs = half_spans[best_j]
            used_half.add(best_j)
            out.append((fs[0], fs[1], (fs[2] + hs[2]) / 2, max(fs[3], hs[3]), "both"))
        else:
            out.append((fs[0], fs[1], fs[2], fs[3], "full_only"))
    for j, hs in enumerate(half_spans):
        if j not in used_half:
            out.append((hs[0], hs[1], hs[2], hs[3], "half_only"))
    out.sort(key=lambda x: (x[0], x[1]))
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

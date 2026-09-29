"""Token windowing helpers.

Extracted from tenzinyonten/layer_detection's common/build_tsawa_dataset.py
(sliding_windows, special_token_ids, unchanged) so this repo does not need
that file's heavier dependencies (pandas, PyYAML, datasets) just for the two
functions detect_layers.py actually calls. pack_window_for_inference is new
here: it is the inference-time counterpart of that file's training-oriented
pack_window, minus the `labels` column (there is nothing to label at
inference time).
"""

from __future__ import annotations

from typing import Any


def sliding_windows(n_tokens: int, content_len: int, stride: int) -> list[tuple[int, int]]:
    """Token-index windows covering [0, n_tokens). Last window is flush-right."""
    if n_tokens <= 0:
        return []
    if n_tokens <= content_len:
        return [(0, n_tokens)]
    spans: list[tuple[int, int]] = []
    start = 0
    while True:
        end = min(start + content_len, n_tokens)
        spans.append((start, end))
        if end == n_tokens:
            break
        start += stride
        if start >= n_tokens:
            break
        # Guarantee the tail is covered even if stride skips past it.
        if start + content_len >= n_tokens and end < n_tokens:
            spans.append((max(0, n_tokens - content_len), n_tokens))
            break
    # Deduplicate if the flush-right window matches the previous one.
    out: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for pair in spans:
        if pair not in seen:
            seen.add(pair)
            out.append(pair)
    return out


def special_token_ids(tokenizer: Any) -> tuple[int, int, int]:
    cls_id = tokenizer.cls_token_id
    if cls_id is None:
        cls_id = tokenizer.bos_token_id
    sep_id = tokenizer.sep_token_id
    if sep_id is None:
        sep_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        # Gemma-style tokenizers sometimes leave pad unset.
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
        pad_id = tokenizer.pad_token_id
    if cls_id is None or sep_id is None or pad_id is None:
        raise SystemExit(
            f"Tokenizer is missing special ids (cls={cls_id}, sep={sep_id}, pad={pad_id})"
        )
    return int(cls_id), int(sep_id), int(pad_id)


def pack_window_for_inference(input_ids, offsets, w_start, w_end, cls_id, sep_id, pad_id, max_length):
    """Add CLS/SEP/pad around one window's content tokens. Returns
    (input_ids, attention_mask, content_offset_mapping)."""
    content_ids = input_ids[w_start:w_end]
    content_off = offsets[w_start:w_end]
    ids = [cls_id] + content_ids + [sep_id]
    mask = [1] * len(ids)
    pad_n = max_length - len(ids)
    if pad_n:
        ids = ids + [pad_id] * pad_n
        mask = mask + [0] * pad_n
    return ids, mask, content_off

"""Command-line argument lists for Stage 2 (infer) and Stage 3 (postprocess).

detect_layers.main() and detect() both run the stages in-process by calling each
stage's main(argv). They build those lists here, in one place, so a new option is
added once. Options left at None / False are not passed, so each stage keeps its
own default.
"""
from __future__ import annotations


def infer_args(input_dir, out_dir, *, layers=None, all_layers=False, device=None, stride=None,
               dual_window=False, author_span="last", title_model=None, author_model=None) -> list[str]:
    argv = ["--input", str(input_dir), "--out", str(out_dir)]
    argv += ["--all"] if all_layers else ["--layers", *layers]
    if device:
        argv += ["--device", device]
    if stride is not None:
        argv += ["--stride", str(stride)]
    if dual_window:
        argv.append("--dual-window")
    argv += ["--author-span", author_span]
    for flag, val in (("--title-model", title_model), ("--author-model", author_model)):
        if val:
            argv += [flag, str(val)]
    return argv


def postprocess_args(input_dir, source_dir, out_dir, *, review_threshold=None,
                     title_author_review_threshold=None, title_author_opf=False,
                     json=False, html=False, opf=False) -> list[str]:
    argv = ["--input", str(input_dir), "--source", str(source_dir), "--out", str(out_dir)]
    if review_threshold is not None:
        argv += ["--review-threshold", str(review_threshold)]
    if title_author_review_threshold is not None:
        argv += ["--title-author-review-threshold", str(title_author_review_threshold)]
    for flag, on in (("--title-author-opf", title_author_opf), ("--json", json),
                     ("--html", html), ("--opf", opf)):
        if on:
            argv.append(flag)
    return argv

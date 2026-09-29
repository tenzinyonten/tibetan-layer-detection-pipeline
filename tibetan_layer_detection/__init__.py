"""tibetan_layer_detection -- detect Tsawa, Sabche, Chapter, Quotation and
Yigchung layers in Tibetan text with per-layer mmBERT models.

Simple API:

    from tibetan_layer_detection import detect
    result = detect("book.txt", layers=["all"])
    result["layers"]["sabche"]   # [{"start": ..., "end": ..., "confidence": ..., "review_needed": ...}, ...]

For the full three-stage pipeline (preprocess / infer / postprocess) with
checkpointing across a large batch, use the console scripts (tibetan-detect,
or the three stages individually: tibetan-preprocess, tibetan-infer,
tibetan-postprocess) instead -- this function runs all three in one process
with no checkpointing, which is right for one book or a handful, not for a
batch you might need to resume.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from . import infer as _infer
from . import postprocess as _postprocess
from . import preprocess as _preprocess
from .layer_config import LAYERS

__all__ = ["detect", "LAYERS"]
__version__ = "0.1.0"

_RESULT_EXCLUDE = {"_batch_summary.json"}


def detect(path, layers="all", *, out_dir=None, device=None, review_threshold: float = 0.7,
          write_json: bool = False, write_html: bool = False, write_opf: bool = False):
    """Run the full three-stage pipeline in-process and return the result(s).

    path: a single text file, or a folder (a mix of .txt files and/or raw
        OpenPecha book folders -- same discovery rule as the CLI's --dir:
        flat *.txt plus any base/v001.txt found at any depth).
    layers: a list of layer names from tibetan_layer_detection.LAYERS
        (e.g. ["tsawa", "sabche"]), or "all" / ["all"] (the default) for
        every configured layer.
    out_dir: where the pipeline's intermediate and final files are written.
        If not given, a temporary directory is used and removed before this
        function returns -- fine when you only want the returned Python
        objects. If you also pass write_json/write_html/write_opf, out_dir
        defaults to "results" in the current directory instead of a
        temp dir, since those files need somewhere to persist.
    device: "cuda" or "cpu"; defaults to cuda if available.
    review_threshold: a span with confidence below this is flagged
        review_needed in the returned data (and in a persisted JSON file,
        if write_json=True).
    write_json / write_html / write_opf: also persist Stage 3's JSON/HTML/
        .opf files under out_dir (see the tibetan-postprocess console
        script's --help for what each one is). A JSON pass always runs
        internally regardless -- it's how this function builds its return
        value -- but the per-book .json files are deleted afterward unless
        write_json=True.

    Returns: one book's result dict ({"book_id", "text_length", "layers",
        and "errors" if there were any}) if `path` is a single file, or a
        list of such dicts (one per book found) if `path` is a directory.
    """
    layer_names = list(LAYERS) if layers in ("all", ["all"]) else list(layers)
    device = device or ("cuda" if _infer._has_cuda() else "cpu")

    use_temp_dir = out_dir is None and not (write_json or write_html or write_opf)
    tmp = None
    try:
        if use_temp_dir:
            tmp = tempfile.TemporaryDirectory(prefix="tibetan_layer_detection_")
            base = Path(tmp.name)
        else:
            base = Path(out_dir) if out_dir is not None else Path("results")
        pre_dir, pred_dir, stage3_dir = base / "preprocessed", base / "predictions", base / "output"

        _preprocess.main(["--input", str(path), "--out", str(pre_dir)])
        _infer.main(["--input", str(pre_dir), "--out", str(pred_dir),
                    "--layers", *layer_names, "--device", device])

        post_argv = ["--input", str(pred_dir), "--source", str(pre_dir), "--out", str(stage3_dir),
                    "--review-threshold", str(review_threshold), "--json"]
        if write_html:
            post_argv.append("--html")
        if write_opf:
            post_argv.append("--opf")
        _postprocess.main(post_argv)

        book_files = [fp for fp in sorted(stage3_dir.glob("*.json"))
                     if fp.name not in _RESULT_EXCLUDE and not fp.name.startswith("run_")]
        results = [json.loads(fp.read_text(encoding="utf-8")) for fp in book_files]
        if not write_json:
            for fp in book_files:
                fp.unlink()
    finally:
        if tmp is not None:
            tmp.cleanup()

    if Path(path).is_file():
        return results[0] if results else None
    return results

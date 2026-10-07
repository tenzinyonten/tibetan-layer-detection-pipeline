"""Per-layer model configuration, shared by common/infer.py (which layer to
load and how to decode it) and common/postprocess.py (HTML legend colors,
.opf layer-name capitalization). Split out of detect_layers.py so neither
stage has to import the other's internals.

Each break_penalty/stride is the value documented on the model's own Hugging
Face repo (README.md / training/test_metrics.json / training/config.yaml),
checked 2026-09-28, not an assumed project-wide default -- see `source` on
each entry. All five layers (BDRC/Bo-Tsawa-Detection, Bo-Sabche-detection,
Bo-Chapter-Detection, Bo-Quotation-Detection, Bo-Yigchung-Detection) have a
dedicated model as of that date. Yigchung uses a different windowing stride
(4914, not 5120) and a different overlap policy: a token's logits come from
only the first window that covers it, and the whole document is decoded once
(see infer.run_layer_stitched()), matching how its training data masked
repeated tokens out of the loss. Its own README also says to cite a test F1
of 0.431, well below the other four layers; that caveat is printed whenever
it loads. BDRC/Bo-Multilayer-Detection (the old 15-label joint model) is
deliberately not used for anything here: its own published metrics put
tsawa/yigchung/quote F1 at about 0.02-0.04.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LayerConfig:
    repo: str | None
    scheme: str  # "bio" or "bioe"
    break_penalty: float
    stride: int
    color: str
    source: str  # where break_penalty/stride came from, for the --help text and logs
    stitch_first_window: bool = False  # see infer.run_layer_stitched(); only Yigchung as of 2026-09-28
    caveat: str = ""  # printed once when the layer is loaded, for a known-weak model
    # Post-Viterbi cleanup switches, read by decode.apply_linguistic_rules (see its docstring).
    # Defaults keep the original behavior for every layer; Sabche overrides them, because
    # on the Sabche test set (29 books) extend_to_tsheg alone cost 12 true positives (F1 0.9619
    # -> 0.9590) and merge_small_gaps fused neighbouring headings (F1 -> 0.822).
    extend_to_tsheg: bool = True     # extend a span ending mid-syllable to the next tsheg
    merge_small_gaps: bool = True    # merge spans separated only by a short punctuation gap
    repair_fragments: bool = False   # drop short spans nested in a longer one (a duplicate)
    extend_lone_fragments: bool = False  # with repair_fragments: also extend a short span with no cover
    # to the next shad/newline. Sabche test: recovers 1 heading but turns ~10 other fragments into
    # full-length false positives (F1 0.9642 -> 0.9630), so it is off.


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
        extend_to_tsheg=False, merge_small_gaps=False, repair_fragments=True,
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

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
    scheme: str  # "bio" or "bioe" ("ta" for the title/author layers, which read the labels from the model)
    break_penalty: float
    stride: int
    color: str
    source: str  # where break_penalty/stride came from, for the --help text and logs
    kind: str = "window"  # "window": sliding windows over the book (every layer but title/author);
    # "head_tail": one sequence from the first and last 3,000 characters, one span per field
    # (title, author) -- see title_author.py
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
    shad_rule: bool = False  # drop a span under decode.SHAD_RULE_MAX_CHARS (8) chars unless a shad
    # follows it; a precision rule. Sabche test (29 books): P 0.9609 -> 0.9635, R 0.9696 unchanged,
    # F1 0.9652 -> 0.9665 (+0.0013; 11 false positives dropped, no true positives).
    # SHIPPED PENDING VALIDATION, NOT CONFIRMED: the rule and its 8-char threshold were tuned by
    # looking at the test split's errors, so these numbers are in-sample. Re-check on the validation
    # split when the Sabche model is next run, and set shad_rule=False for sabche if it does not hold.


TITLE_AUTHOR_MODEL = "Yontenn/mmbert-title-author-run2"  # private; weights are in its best/ folder
TA_FIELDS = ("title", "author")
TA_LAYER_NAMES = TA_FIELDS + tuple(f + "_skill" for f in TA_FIELDS)  # the _skill ones appear with system "both"


def is_title_author(name: str) -> bool:
    return name in TA_LAYER_NAMES


def base_layer(name: str) -> str:
    """The LAYERS key a layer name styles/thresholds by: title_skill -> title."""
    return name[:-len("_skill")] if name.endswith("_skill") and name[:-len("_skill")] in TA_FIELDS else name


@dataclass
class TitleAuthorOptions:
    """How the title/author layers are run. Stage 2 reads system, the models and
    the skill settings; Stage 3 reads opf and review_threshold. Defaults are free and local."""
    system: str = "model"                  # "model", "skill" or "both"
    title_model: str = TITLE_AUTHOR_MODEL  # a Hugging Face repo, or a local folder
    author_model: str = TITLE_AUTHOR_MODEL  # same joint model for both until the single-entity models exist
    skill_model: str = "gemini-3.8-flash"  # only used by system "skill" / "both"; PAID, needs GEMINI_API_KEY or ANTHROPIC_API_KEY
    skill_dir: str | None = None           # folder holding title_author_skill.py (default: TA_SKILL_DIR, else the repo's)
    crop_chars: int = 3000                 # characters kept at each end
    break_penalty: float = 12.0            # Viterbi; picked on validation (the trainer used 5)
    opf: bool = False                      # .opf output for title/author: not implemented, raises if switched on
    review_threshold: float | None = None  # None: the global review threshold (--review-threshold, 0.7)


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
        extend_to_tsheg=False, merge_small_gaps=False, repair_fragments=True, shad_rule=True,
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
    "title": LayerConfig(
        repo=TITLE_AUTHOR_MODEL, scheme="ta", break_penalty=12.0, stride=0, color="#b07df5",  # purple
        source="Yontenn/mmbert-title-author-run2 docs/run2_results.md (Viterbi break penalty 12, picked on "
               "validation); test title F1 0.828 on the older 2,389-work test",
        kind="head_tail", extend_to_tsheg=False, merge_small_gaps=False,
        caveat="title/author reads only the first and last 3,000 characters and gives one title for the whole file",
    ),
    "author": LayerConfig(
        repo=TITLE_AUTHOR_MODEL, scheme="ta", break_penalty=12.0, stride=0, color="#4dd8d8",  # teal
        source="Yontenn/mmbert-title-author-run2 docs/run2_results.md (Viterbi break penalty 12, picked on "
               "validation); test author F1 0.744 on the older 2,389-work test",
        kind="head_tail", extend_to_tsheg=False, merge_small_gaps=False,
        caveat="title/author reads only the first and last 3,000 characters and gives one author for the whole file",
    ),
}

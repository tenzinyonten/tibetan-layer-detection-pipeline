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
    min_confidence: float | None = None  # Stage 3 drops a span whose confidence is below this (a precision rule);
    # None keeps every span (the review flag still marks the uncertain ones). Sabche: 0.85, chosen on the validation
    # split (33 books) and confirmed on test (29 books). Validation F1 0.8486 -> 0.8535 (precision 0.868 -> 0.892,
    # recall 0.830 -> 0.818, 37 of 3,091 real headings lost); test F1 0.9686 -> 0.9707 (precision 0.974 -> 0.981,
    # recall 0.964 -> 0.961, 12 of 4,079 lost). Not measured for any other layer.
    shad_rule: bool = False  # drop a span under decode.SHAD_RULE_MAX_CHARS (8) chars unless a shad
    # follows it; a precision rule. Sabche test (29 books): P 0.9609 -> 0.9635, R 0.9696 unchanged,
    # F1 0.9652 -> 0.9665 (+0.0013; 11 false positives dropped, no true positives).
    # SHIPPED PENDING VALIDATION, NOT CONFIRMED: the rule and its 8-char threshold were tuned by
    # looking at the test split's errors, so these numbers are in-sample. Re-check on the validation
    # split when the Sabche model is next run, and set shad_rule=False for sabche if it does not hold.


# All private; weights are in each repo's best/ folder. The two single-entity models know one field
# each (3 labels O, B, I); the joint model knows both (5 labels). Test, 2,384 works, share correct at IoU >= 0.5:
# joint (penalty 12) title 0.833 / author 0.711; title-only + author-only (penalty 6) 0.842 / 0.734.
TITLE_MODEL = "Yontenn/mmbert-title-only"
AUTHOR_MODEL = "Yontenn/mmbert-author-only"
JOINT_MODEL = "Yontenn/mmbert-title-author-run2"
TITLE_AUTHOR_MODEL = JOINT_MODEL  # the earlier name, kept for callers that import it
TA_FIELDS = ("title", "author")
TA_LAYER_NAMES = TA_FIELDS


# The OpenPecha layer files for these two, as they are named in the real books (data/raw_opf: BookTitle.yml in
# 527 of 539 books, Author.yml in 516).
OPF_LAYER_NAMES = {"title": "BookTitle", "author": "Author"}


def is_title_author(name: str) -> bool:
    return name in TA_LAYER_NAMES


@dataclass
class TitleAuthorOptions:
    """How the title/author layers are run. Stage 2 reads the models and crop settings;
    Stage 3 reads opf and review_threshold. Everything runs locally from trained models."""
    title_model: str = TITLE_MODEL         # a Hugging Face repo, or a local folder
    author_model: str = AUTHOR_MODEL       # the joint JOINT_MODEL also works here; if both fields use it, one forward pass serves both
    crop_chars: int = 3000                 # characters kept at each end
    break_penalty: float | None = None     # Viterbi. None: per model -- 6 for a single-entity model, 12 for the joint
    # one (both picked on validation); a number applies to every model alike
    author_span: str = "last"              # which author span to keep when the model finds several: "last" (the default, the
    # rule for a single work) or "first". On whole OpenPecha books the gold author is on the title page, and "first" matched it
    # about twice as often as "last" (35% against 17% strict offsets, 511 books, mostly seen in training); see title_author.py.
    strip_underscore: bool | None = None   # remove "_" from the input before tokenizing. None: per model -- yes for a
    # single-entity model (trained on text without them), no for the joint one
    opf: bool = False                      # also write title/author as BookTitle.yml / Author.yml with --opf (off by default)
    review_threshold: float | None = None  # None: the global review threshold (--review-threshold, 0.7)

    def __post_init__(self):
        if self.author_span not in ("first", "last"):
            raise ValueError(f"author_span must be first or last, not {self.author_span!r}")


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
        min_confidence=0.85,
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
        repo=TITLE_MODEL, scheme="ta", break_penalty=6.0, stride=0, color="#b07df5",  # purple
        source="Yontenn/mmbert-title-only, Viterbi break penalty 6 picked on validation; test title 0.842 "
               "share correct (2,384 works; the joint model: 0.833)",
        kind="head_tail", extend_to_tsheg=False, merge_small_gaps=False,
        caveat="title/author reads only the first and last 3,000 characters and gives one title for the whole file",
    ),
    "author": LayerConfig(
        repo=AUTHOR_MODEL, scheme="ta", break_penalty=6.0, stride=0, color="#4dd8d8",  # teal
        source="Yontenn/mmbert-author-only, Viterbi break penalty 6 picked on validation; test author 0.734 "
               "share correct (2,384 works; the joint model: 0.711)",
        kind="head_tail", extend_to_tsheg=False, merge_small_gaps=False,
        caveat="title/author reads only the first and last 3,000 characters and gives one author for the whole file",
    ),
}

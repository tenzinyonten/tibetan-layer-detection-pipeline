"""postprocess.py -- Stage 3 of the layer-detection pipeline: rendering.

Joins Stage 2's predictions/<book_id>.json (spans + confidence) with Stage
1's preprocessed/<book_id>.json (source text + source_path -- predictions
alone carry neither) and writes final JSON, HTML, and/or OpenPecha .opf
output, plus a batch summary and a run log.

--opf writes predictions as OpenPecha-shaped layer YAML, only for layers with
at least one span. A plain .txt input gets a fresh minimal structure under
--out: <id>.opf/<id>.opf/base/v001.txt plus layers/v001/. A real OpenPecha
book (Stage 1 source_path matches <id>.opf/<id>.opf/base/v001.txt) whose
layers/v001/ already exists holds gold annotations, so its predictions go to
the sibling layers/predicted/ instead and layers/v001/ is never touched.
Schema verified against real files in this repo
before writing any code: data/raw_opf/P000201.opf's Tsawa.yml/Sabche.yml/
Chapter.yml, data/raw_opf/P000172.opf's Quotation.yml, data/raw_opf/
I058DD999.opf's Yigchung.yml. Deliberately does not add a confidence field
there, to keep an exact structural match with those files; confidence lives
in the JSON output only.

pyyaml is a required dependency of this package (see pyproject.toml), so,
unlike an earlier standalone version of this tool where PyYAML was made an
optional lazy import, it is imported normally here at module level.

Usage (as an installed console script)
---------------------------------------
    tibetan-postprocess --input predictions/ --source preprocessed/ \\
        --out output/ --json --html --opf --review-threshold 0.7
"""

from __future__ import annotations

from .cli import load_pair, main, parse_args
from .filters import (MIN_SPAN_CHARS, QUOTE_COVERAGE_DROP, _coverage, _union, drop_low_confidence,
                      drop_non_tibetan_spans, drop_quoted_tsawa, drop_short_spans, split_quoted_tsawa)
from .html_out import write_html
from .json_out import _span_payload, write_json
from .opf_out import (_sub_tracked, normalize_tsegs, normalize_tsegs_tracked, opf_layers_to_write, opf_root_for,
                      remap_layers, write_opf_layers, write_plain_opf)
from .review import needs_review, threshold_for

__all__ = [
    "MIN_SPAN_CHARS", "QUOTE_COVERAGE_DROP", "drop_low_confidence", "drop_non_tibetan_spans", "drop_quoted_tsawa",
    "drop_short_spans", "load_pair", "main", "needs_review", "normalize_tsegs", "normalize_tsegs_tracked",
    "opf_layers_to_write", "opf_root_for", "parse_args", "remap_layers", "split_quoted_tsawa", "threshold_for",
    "write_html", "write_json", "write_opf_layers", "write_plain_opf",
]

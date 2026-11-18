"""Stage 3 OpenPecha output (--opf): layer YAML, plain .txt input, text normalisation with offset remapping."""

from __future__ import annotations

import re
import uuid
from bisect import bisect_left
from pathlib import Path

import yaml

from ..layer_config import OPF_LAYER_NAMES, is_title_author


def opf_root_for(source_path: Path) -> Path | None:
    """<id>.opf/<id>.opf/base/v001.txt -> the inner <id>.opf directory (the
    real OpenPecha root, where layers/v001/ lives), or None if source_path
    doesn't match that exact layout."""
    if source_path.name == "v001.txt" and source_path.parent.name == "base":
        grandparent = source_path.parent.parent
        if grandparent.name.endswith(".opf"):
            return grandparent
    return None


def normalize_tsegs(text: str) -> str:
    text = re.sub(r'\s+་', '་ ', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()


def _sub_tracked(pattern: str, new: str, text: str, src: list[int], pick) -> tuple[str, list[int]]:
    out, out_src, pos = [], [], 0
    for m in re.finditer(pattern, text):
        out.append(text[pos:m.start()])
        out_src.extend(src[pos:m.start()])
        out.append(new)
        out_src.extend(pick(m, src))
        pos = m.end()
    out.append(text[pos:])
    out_src.extend(src[pos:])
    return "".join(out), out_src


def normalize_tsegs_tracked(text: str) -> tuple[str, list[int]]:
    """Same result as normalize_tsegs, plus for every output character the
    offset of the input character it came from (non-decreasing)."""
    src = list(range(len(text)))
    text, src = _sub_tracked(r'\s+་', '་ ', text, src,
                             lambda m, s: [s[m.start()], s[m.end() - 1]])
    text, src = _sub_tracked(r' +', ' ', text, src, lambda m, s: [s[m.start()]])
    lead = len(text) - len(text.lstrip())
    text = text.strip()
    return text, src[lead:lead + len(text)]


def remap_layers(layers: dict[str, list[dict]], src: list[int]) -> dict[str, list[dict]]:
    """Move span offsets from the original text to the normalized one. A span
    that only covered removed whitespace disappears."""
    out = {}
    for name, spans in layers.items():
        kept = []
        for s in spans:
            start, end = bisect_left(src, s["start"]), bisect_left(src, s["end"])
            if end > start:
                kept.append({**s, "start": start, "end": end})
        out[name] = kept
    return out


def write_plain_opf(book_id: str, text: str, out_dir: Path) -> Path:
    """For a plain .txt input: create <out>/<id>.opf/<id>.opf/ with
    base/v001.txt (the text given) and return that inner root, ready for
    write_opf_layers (which fills layers/v001/ via write_opf_layers)."""
    root = out_dir / f"{book_id}.opf" / f"{book_id}.opf"
    (root / "base").mkdir(parents=True, exist_ok=True)
    (root / "base" / "v001.txt").write_text(text, encoding="utf-8")
    return root


def opf_layers_to_write(layers: dict[str, list[dict]], include_title_author: bool) -> dict[str, list[dict]]:
    """The layers that go to .opf: every layer except title/author, which are added (as the
    title and author) only when include_title_author is set."""
    return {n: sp for n, sp in layers.items()
            if not is_title_author(n) or (include_title_author and n in OPF_LAYER_NAMES)}


def write_opf_layers(layers: dict[str, list[dict]], opf_root: Path,
                     subdir: str = "v001") -> list[Path]:
    """Write predictions as OpenPecha-shaped layer YAML under
    layers/<subdir>/, skipping layers with no spans (see module docstring for
    which subdir is used when, and the real files this schema was verified
    against)."""
    out_dir = opf_root / "layers" / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, spans in layers.items():
        if not spans:
            continue
        layer_name = OPF_LAYER_NAMES.get(name, name.capitalize())
        data = {
            "id": uuid.uuid4().hex,
            "annotation_type": layer_name,
            "revision": "00001",
            "annotations": {
                uuid.uuid4().hex: {"span": {"start": int(s["start"]), "end": int(s["end"])}}
                for s in spans
            },
        }
        p = out_dir / f"{layer_name}.yml"
        p.write_text(yaml.safe_dump(data, default_flow_style=False, sort_keys=False,
                                    allow_unicode=True), encoding="utf-8")
        written.append(p)
    return written

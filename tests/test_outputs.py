"""Stage 3 writers and text helpers: HTML, OpenPecha .opf pieces, JSON, text normalisation.
These pin the behaviour of postprocess before it was split into modules."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from tibetan_layer_detection import postprocess as P  # noqa: E402

TEXT = "ཀ  ་ཁ   ག "


def span(s, e, c=0.9, **kw):
    return {"start": s, "end": e, "confidence": c, "window_votes": 1, **kw}


class Normalise(unittest.TestCase):
    def test_normalize_tsegs(self):
        self.assertEqual(P.normalize_tsegs(TEXT), "ཀ་ ཁ ག")

    def test_tracked_gives_the_same_text_and_a_source_offset_per_character(self):
        out, src = P.normalize_tsegs_tracked(TEXT)
        self.assertEqual(out, P.normalize_tsegs(TEXT))
        self.assertEqual(len(src), len(out))
        self.assertEqual(src, sorted(src))
        self.assertEqual(src, [0, 1, 3, 4, 5, 8])

    def test_remap_moves_offsets_and_drops_a_span_that_only_covered_removed_whitespace(self):
        _, src = P.normalize_tsegs_tracked(TEXT)
        got = P.remap_layers({"tsawa": [span(0, 1), span(1, 3), span(4, 6, None)]}, src)["tsawa"]
        self.assertEqual([(s["start"], s["end"]) for s in got], [(0, 1), (1, 2), (3, 5)])
        self.assertEqual(P.remap_layers({"tsawa": [span(5, 7)]}, [0, 5, 6, 7])["tsawa"][0]["start"], 1)  # kept, moved
        self.assertEqual(P.remap_layers({"tsawa": [span(1, 3)]}, [0, 5, 6, 7])["tsawa"], [])  # only removed characters


class Opf(unittest.TestCase):
    def test_opf_root_for(self):
        self.assertEqual(P.opf_root_for(Path("/d/I1.opf/I1.opf/base/v001.txt")), Path("/d/I1.opf/I1.opf"))
        self.assertIsNone(P.opf_root_for(Path("/d/book.txt")))
        self.assertIsNone(P.opf_root_for(Path("/d/I1.opf/I1.opf/base/other.txt")))

    def test_write_plain_opf_and_layers(self):
        with tempfile.TemporaryDirectory() as d:
            root = P.write_plain_opf("b1", "བཀྲ་ཤིས", Path(d))
            self.assertEqual(root, Path(d) / "b1.opf" / "b1.opf")
            self.assertEqual((root / "base" / "v001.txt").read_text(encoding="utf-8"), "བཀྲ་ཤིས")
            written = P.write_opf_layers({"tsawa": [span(0, 4)], "sabche": [], "title": [span(0, 3)]}, root)
            self.assertEqual(sorted(p.name for p in written), ["BookTitle.yml", "Tsawa.yml"])  # empty sabche skipped
            data = yaml.safe_load((root / "layers" / "v001" / "Tsawa.yml").read_text(encoding="utf-8"))
            self.assertEqual((data["annotation_type"], data["revision"]), ("Tsawa", "00001"))
            (ann,) = data["annotations"].values()
            self.assertEqual(ann, {"span": {"start": 0, "end": 4}})
            P.write_opf_layers({"tsawa": [span(0, 4)]}, root, subdir="predicted")
            self.assertTrue((root / "layers" / "predicted" / "Tsawa.yml").is_file())


class Json(unittest.TestCase):
    def test_write_json_flags_review_and_keeps_errors(self):
        with tempfile.TemporaryDirectory() as d:
            p = P.write_json("b1", 100, {"tsawa": [span(0, 9, 0.99), span(10, 19, 0.2)]}, {"sabche": "boom"}, Path(d), 0.7)
            got = json.loads(p.read_text(encoding="utf-8"))
        self.assertEqual((got["book_id"], got["text_length"], got["errors"]), ("b1", 100, {"sabche": "boom"}))
        self.assertEqual([s["review_needed"] for s in got["layers"]["tsawa"]], [False, True])
        self.assertNotIn("window_votes", got["layers"]["tsawa"][0])

    def test_load_pair_returns_none_when_stage_1_file_is_missing(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "b1.json").write_text(json.dumps({"book_id": "b1", "layers": {}}), encoding="utf-8")
            pred, pre = P.load_pair("b1", d, d / "nowhere")
        self.assertEqual(pred["book_id"], "b1")
        self.assertIsNone(pre)


class Html(unittest.TestCase):
    def test_html_has_legend_counts_errors_and_marks_spans(self):
        text = "བཀྲ་ཤིས་བདེ་ལེགས། <b>x</b>"
        with tempfile.TemporaryDirectory() as d:
            p = P.write_html("book<1>", text, {"tsawa": [span(0, 8)], "sabche": []}, {"chapter": "model load failed"}, Path(d))
            page = p.read_text(encoding="utf-8")
        self.assertEqual(p.name, "book<1>.html")
        self.assertIn("book&lt;1&gt;", page)                      # the id is escaped
        self.assertIn("tsawa <span class=\"mut\">(1 spans)", page)
        self.assertIn("sabche <span class=\"mut\">(0 spans)", page)
        self.assertIn("chapter <span class=\"mut\">(0 spans, model load failed)", page)
        self.assertIn('class="span"', page)
        self.assertIn('title="tsawa"', page)
        self.assertIn("&lt;b&gt;x&lt;/b&gt;", page)                # the text is escaped
        self.assertEqual(page.count('class="span"'), 1)

    def test_overlapping_layers_stack_underlines(self):
        with tempfile.TemporaryDirectory() as d:
            page = P.write_html("b", "ཀཁགངཅཆ", {"tsawa": [span(0, 5)], "sabche": [span(2, 7)]}, {}, Path(d)).read_text(encoding="utf-8")
        self.assertIn('title="sabche, tsawa"', page)              # the overlap carries both layers


if __name__ == "__main__":
    unittest.main()

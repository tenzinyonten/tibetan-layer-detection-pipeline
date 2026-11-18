"""Stage 3 low-confidence cutoff (LayerConfig.min_confidence; Sabche 0.85)."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tibetan_layer_detection import postprocess as P  # noqa: E402
from tibetan_layer_detection.layer_config import LAYERS  # noqa: E402


def sp(c, s=0, e=40):
    return {"start": s, "end": e, "confidence": c, "window_votes": 1}


class MinConfidence(unittest.TestCase):
    def test_only_sabche_has_a_cutoff(self):
        self.assertEqual({n: c.min_confidence for n, c in LAYERS.items() if c.min_confidence is not None}, {"sabche": 0.85})

    def test_drops_below_the_cutoff_and_keeps_equal(self):
        layers = {"sabche": [sp(0.99), sp(0.85), sp(0.8499), sp(0.2)]}
        out, dropped = P.drop_low_confidence(layers)
        self.assertEqual([s["confidence"] for s in out["sabche"]], [0.99, 0.85])
        self.assertEqual([s["confidence"] for s in dropped["sabche"]], [0.8499, 0.2])

    def test_other_layers_and_title_author_are_untouched(self):
        layers = {"tsawa": [sp(0.1)], "chapter": [sp(0.1)], "title": [sp(0.1)], "author": [sp(0.1)]}
        out, dropped = P.drop_low_confidence(layers)
        self.assertEqual(out, layers)
        self.assertEqual(dropped, {})

    def test_a_span_without_confidence_is_never_dropped(self):
        out, dropped = P.drop_low_confidence({"sabche": [sp(None), sp(0.1)]})
        self.assertEqual(len(out["sabche"]), 1)
        self.assertIsNone(out["sabche"][0]["confidence"])

    def test_nothing_dropped_leaves_layers_equal(self):
        layers = {"sabche": [sp(0.9)]}
        out, dropped = P.drop_low_confidence(layers)
        self.assertEqual((out, dropped), (layers, {}))

    def _run(self, *extra):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "pre").mkdir(); (d / "pred").mkdir()
            body = "བཀྲ་ཤིས་བདེ་ལེགས། " * 20
            (d / "pre" / "b.json").write_text(json.dumps(
                {"book_id": "b", "text": body, "char_count": len(body), "source_path": str(d / "b.txt")}), encoding="utf-8")
            (d / "pred" / "b.json").write_text(json.dumps({"book_id": "b", "layers": {
                "sabche": [sp(0.95, 0, 40), sp(0.5, 50, 90)], "tsawa": [sp(0.3, 100, 140)]}}), encoding="utf-8")
            P.main(["--input", str(d / "pred"), "--source", str(d / "pre"), "--out", str(d / "out"), "--json", *extra])
            res = json.loads((d / "out" / "b.json").read_text(encoding="utf-8"))
            summ = json.loads((d / "out" / "_batch_summary.json").read_text(encoding="utf-8"))[0]
        return res, summ

    def test_stage_3_drops_and_lists_the_dropped_spans(self):
        res, summ = self._run()
        self.assertEqual({k: len(v) for k, v in res["layers"].items()}, {"sabche": 1, "tsawa": 1})
        self.assertEqual(summ["low_confidence_dropped"], {"sabche": 1})
        self.assertEqual([(s["start"], s["end"], s["confidence"]) for s in summ["low_confidence_dropped_spans"]["sabche"]], [(50, 90, 0.5)])

    def test_keep_low_confidence_switches_it_off(self):
        res, summ = self._run("--keep-low-confidence")
        self.assertEqual(len(res["layers"]["sabche"]), 2)
        self.assertEqual(summ["low_confidence_dropped"], {})


if __name__ == "__main__":
    unittest.main()

"""Unit tests for Stage 3 drop_quoted_tsawa (predicted quotation, not gold)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tibetan_layer_detection.infer import layers_for_run  # noqa: E402
from tibetan_layer_detection.postprocess import drop_quoted_tsawa, split_quoted_tsawa  # noqa: E402


def span(start, end, confidence=1.0):
    return {"start": start, "end": end, "confidence": confidence}


class TestDropQuotedTsawa(unittest.TestCase):
    def test_drops_when_half_covered(self):
        layers = {
            "tsawa": [span(0, 10), span(20, 30)],
            "quotation": [span(0, 8)],
        }
        out, n = drop_quoted_tsawa(layers)
        self.assertEqual(n, 1)
        self.assertEqual([(s["start"], s["end"]) for s in out["tsawa"]], [(20, 30)])
        self.assertEqual(len(out["quotation"]), 1)

    def test_keeps_partial_overlap_under_threshold(self):
        layers = {
            "tsawa": [span(0, 10)],
            "quotation": [span(0, 4)],  # 40%
        }
        out, n = drop_quoted_tsawa(layers)
        self.assertEqual(n, 0)
        self.assertEqual(len(out["tsawa"]), 1)

    def test_noop_without_quotation_layer(self):
        layers = {"tsawa": [span(0, 10)]}
        out, n = drop_quoted_tsawa(layers)
        self.assertEqual(n, 0)
        self.assertIs(out, layers)

    def test_noop_when_quotation_model_found_nothing(self):
        layers = {"tsawa": [span(0, 10)], "quotation": []}
        out, n = drop_quoted_tsawa(layers)
        self.assertEqual(n, 0)
        self.assertEqual(out["tsawa"], layers["tsawa"])

    def test_ignores_a_gold_key_if_someone_passed_one(self):
        layers = {
            "tsawa": [span(0, 10)],
            "quotation": [],
            "gold_quotation": [span(0, 10)],
        }
        out, n = drop_quoted_tsawa(layers)
        self.assertEqual(n, 0)
        self.assertEqual(len(out["tsawa"]), 1)

    def test_noop_without_tsawa(self):
        layers = {"quotation": [span(0, 10)]}
        out, n = drop_quoted_tsawa(layers)
        self.assertEqual(n, 0)
        self.assertIs(out, layers)


class TestCoverageBoundaryAndUnion(unittest.TestCase):
    def kept(self, tsawa, quotes):
        out, n = drop_quoted_tsawa({"tsawa": [tsawa], "quotation": quotes})
        return n == 0

    def test_boundary_coverage(self):
        t = span(0, 100)
        self.assertTrue(self.kept(t, [span(0, 1)]))          # 0.01
        self.assertTrue(self.kept(t, [span(200, 300)]))      # 0.0, disjoint
        self.assertTrue(self.kept(t, [span(0, 49)]))         # 0.49 kept
        self.assertFalse(self.kept(t, [span(0, 50)]))        # 0.50 dropped
        self.assertFalse(self.kept(t, [span(0, 100)]))       # 1.0 dropped

    def test_overlapping_quotations_not_double_counted(self):
        # union is 0-30 (30%); summing the two overlaps would give 60% and wrongly drop it
        self.assertTrue(self.kept(span(0, 100), [span(0, 30), span(0, 30)]))
        self.assertTrue(self.kept(span(0, 100), [span(0, 30), span(10, 30)]))

    def test_adjacent_quotations_add_up(self):
        self.assertFalse(self.kept(span(0, 100), [span(0, 25), span(25, 50)]))  # union 0-50

    def test_quotation_layer_is_never_changed(self):
        quotes = [span(0, 8), span(5, 9)]
        out, dropped = split_quoted_tsawa({"tsawa": [span(0, 10)], "quotation": quotes})
        self.assertEqual(out["quotation"], quotes)
        self.assertEqual(len(dropped), 1)

    def test_dropped_spans_are_returned_with_coverage(self):
        layers = {"tsawa": [span(0, 10, 0.9), span(20, 30, 0.8)], "quotation": [span(0, 8)]}
        out, dropped = split_quoted_tsawa(layers)
        self.assertEqual(dropped, [{"start": 0, "end": 10, "confidence": 0.9, "quote_coverage": 0.8}])
        self.assertEqual([(s["start"], s["end"]) for s in out["tsawa"]], [(20, 30)])

    def test_nothing_dropped_returns_same_object(self):
        layers = {"tsawa": [span(0, 10)], "quotation": [span(50, 60)]}
        out, dropped = split_quoted_tsawa(layers)
        self.assertIs(out, layers)
        self.assertEqual(dropped, [])


class TestLayersForRun(unittest.TestCase):
    def test_tsawa_only_also_runs_quotation_model(self):
        self.assertEqual(layers_for_run(["tsawa"]), ["tsawa", "quotation"])

    def test_skip_quotation(self):
        self.assertEqual(layers_for_run(["tsawa"], skip_quotation=True), ["tsawa"])

    def test_sabche_only_unchanged(self):
        self.assertEqual(layers_for_run(["sabche"]), ["sabche"])


if __name__ == "__main__":
    unittest.main()

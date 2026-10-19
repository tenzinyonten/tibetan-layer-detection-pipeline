import unittest

from tibetan_layer_detection.decode import apply_linguistic_rules, drop_non_tibetan, has_tibetan
from tibetan_layer_detection.postprocess import drop_non_tibetan_spans

ENGLISH = "Return to the table of contents"
TIBETAN = "བཀྲ་ཤིས་བདེ་ལེགས།"
TEXT = f"{ENGLISH} {TIBETAN} 123"
ENG = (0, len(ENGLISH))
TIB = (len(ENGLISH) + 1, len(ENGLISH) + 1 + len(TIBETAN))


class HasTibetan(unittest.TestCase):
    def test_cases(self):
        self.assertTrue(has_tibetan(TIBETAN))
        self.assertTrue(has_tibetan("abc བ"))
        self.assertFalse(has_tibetan(ENGLISH))
        self.assertFalse(has_tibetan(" 123 \n"))
        self.assertFalse(has_tibetan(""))


class DropNonTibetanSpans(unittest.TestCase):
    def layers(self):
        return {
            "sabche": [{"start": ENG[0], "end": ENG[1], "confidence": 0.9},
                       {"start": TIB[0], "end": TIB[1], "confidence": 0.9}],
            "tsawa": [{"start": TIB[0], "end": TIB[1], "confidence": 0.8}],
            "chapter": [],
        }

    def test_drops_english_only_in_every_layer(self):
        kept, n = drop_non_tibetan_spans(self.layers(), TEXT)
        self.assertEqual(n, {"sabche": 1, "tsawa": 0, "chapter": 0})
        self.assertEqual([(s["start"], s["end"]) for s in kept["sabche"]], [TIB])
        self.assertEqual(len(kept["tsawa"]), 1)
        self.assertEqual(kept["chapter"], [])

    def test_keeps_mixed_span(self):
        mixed = {"sabche": [{"start": 0, "end": len(TEXT), "confidence": 0.9}]}
        kept, n = drop_non_tibetan_spans(mixed, TEXT)
        self.assertEqual(n, {"sabche": 0})
        self.assertEqual(len(kept["sabche"]), 1)

    def test_does_not_mutate_input(self):
        layers = self.layers()
        drop_non_tibetan_spans(layers, TEXT)
        self.assertEqual(len(layers["sabche"]), 2)


class DecodeStage(unittest.TestCase):
    def test_drop_non_tibetan(self):
        spans = [(*ENG, 0.9, 1), (*TIB, 0.9, 1)]
        kept, n = drop_non_tibetan(spans, TEXT)
        self.assertEqual((kept, n), ([(*TIB, 0.9, 1)], 1))

    def test_apply_linguistic_rules_counts_and_drops(self):
        spans = [(*ENG, 0.9, 1), (*TIB, 0.9, 1)]
        kept, n = apply_linguistic_rules(spans, TEXT, extend_to_tsheg=False, merge_gaps=False)
        self.assertEqual(kept, [(*TIB, 0.9, 1)])
        self.assertEqual(n, 1)


if __name__ == "__main__":
    unittest.main()

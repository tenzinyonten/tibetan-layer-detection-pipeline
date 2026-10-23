import unittest

from tibetan_layer_detection.decode import apply_linguistic_rules, drop_short_unless_shad
from tibetan_layer_detection.layer_config import LAYERS

#        0         1         2
TEXT = "ལྔ་པ་ནི།\nགཉིས་པ་རྡོ་རྗེའི་ཆོ་ག་ནི།\nག" + "ཉིས་པ་ཡིན།"


def sp(s, e):
    return (s, e, 0.9, 1)


class ShadRule(unittest.TestCase):
    def test_short_span_kept_only_before_shad(self):
        i = TEXT.index("།")
        followed = sp(0, i)                       # "ལྔ་པ་ནི" then །
        j = TEXT.rindex("ག")
        cut = sp(j, j + 1)                        # "ག" then a letter
        kept, n = drop_short_unless_shad([followed, cut], TEXT)
        self.assertEqual((kept, n), ([followed], 1))

    def test_long_span_untouched(self):
        j = TEXT.index("གཉིས་པ")
        long_span = sp(j, j + 9)                  # >= 8 chars, not followed by a shad
        self.assertNotEqual(TEXT[long_span[1]], "།")
        kept, n = drop_short_unless_shad([long_span], TEXT)
        self.assertEqual((kept, n), ([long_span], 0))

    def test_span_at_end_of_text(self):
        kept, n = drop_short_unless_shad([sp(len(TEXT) - 3, len(TEXT))], TEXT)
        self.assertEqual((kept, n), ([], 1))

    def test_off_by_default_and_on_for_sabche_only(self):
        j = TEXT.rindex("ག")
        cut = sp(j, j + 6)                        # 6 chars, 5 <= len < 8, not followed by a shad
        self.assertEqual(TEXT[cut[1]:cut[1] + 1] == "།", False)
        kept, _ = apply_linguistic_rules([cut], TEXT, extend_to_tsheg=False, merge_gaps=False)
        self.assertEqual(kept, [cut])
        kept, n = apply_linguistic_rules([cut], TEXT, extend_to_tsheg=False, merge_gaps=False, shad_rule=True)
        self.assertEqual((kept, n), ([], 1))
        self.assertEqual({k for k, c in LAYERS.items() if c.shad_rule}, {"sabche"})


if __name__ == "__main__":
    unittest.main()

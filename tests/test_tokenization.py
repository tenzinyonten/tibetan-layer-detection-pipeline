"""A book is tokenized once and shared by every layer whose tokenizer is identical. Offline, no models."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tibetan_layer_detection import infer  # noqa: E402


class FakeTok:
    """Counts how often it tokenizes; `rules` stands for the whole tokenizer definition."""
    is_fast = True

    def __init__(self, rules: str):
        self.rules, self.calls = rules, 0
        self.backend_tokenizer = self

    def to_str(self):
        return self.rules

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=True):
        self.calls += 1
        return {"input_ids": list(range(len(text))), "offset_mapping": [(i, i + 1) for i in range(len(text))]}


class EncodeText(unittest.TestCase):
    def test_identical_tokenizers_share_one_tokenization(self):
        a, b = FakeTok("mmBERT"), FakeTok("mmBERT")
        cache = {}
        e1, e2 = infer.encode_text("abc", a, cache), infer.encode_text("abc", b, cache)
        self.assertIs(e1, e2)
        self.assertEqual((a.calls, b.calls), (1, 0))

    def test_a_different_tokenizer_tokenizes_for_itself(self):
        a, b = FakeTok("mmBERT"), FakeTok("another vocabulary")
        cache = {}
        infer.encode_text("abc", a, cache)
        infer.encode_text("abc", b, cache)
        self.assertEqual((a.calls, b.calls), (1, 1))

    def test_without_a_cache_it_always_tokenizes(self):
        a = FakeTok("mmBERT")
        infer.encode_text("abc", a)
        infer.encode_text("abc", a)
        self.assertEqual(a.calls, 2)


class PerBook(unittest.TestCase):
    def setUp(self):
        self._load, self._run, self._dual = infer.load_layer_model, infer.run_layer, infer.run_layer_dual
        self.seen = []
        self.tok = FakeTok("mmBERT")
        infer.load_layer_model = lambda name, device: (self.tok, object())

        def fake_run(text, tok, model, cfg, device, stride=None, content_max=None, enc=None):
            self.seen.append(enc if enc is not None else tok(text))
            return [], 1
        infer.run_layer = fake_run

    def tearDown(self):
        infer.load_layer_model, infer.run_layer, infer.run_layer_dual = self._load, self._run, self._dual

    def test_one_book_through_three_layers_is_tokenized_once(self):
        infer.infer_one({"book_id": "b", "text": "abcdef"}, ["tsawa", "sabche", "chapter"], "cpu")
        self.assertEqual(self.tok.calls, 1)
        self.assertEqual(len(self.seen), 3)
        self.assertTrue(all(e is self.seen[0] for e in self.seen))

    def test_the_next_book_is_tokenized_again(self):
        infer.infer_one({"book_id": "a", "text": "abcdef"}, ["tsawa", "sabche"], "cpu")
        infer.infer_one({"book_id": "b", "text": "ghijkl"}, ["tsawa", "sabche"], "cpu")
        self.assertEqual(self.tok.calls, 2)

    def test_dual_window_tokenizes_once_for_its_two_passes(self):
        infer.run_layer_dual("abcdef", self.tok, object(), infer.LAYERS["tsawa"], "cpu")
        self.assertEqual(self.tok.calls, 1)
        self.assertEqual(len(self.seen), 2)
        self.assertIs(self.seen[0], self.seen[1])


if __name__ == "__main__":
    unittest.main()

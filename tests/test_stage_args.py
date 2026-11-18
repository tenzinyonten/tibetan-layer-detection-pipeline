"""The Stage 2 / Stage 3 argument lists that detect_layers.main() and detect() build.
The expected lists are what the two callers built inline before they shared stage_args."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tibetan_layer_detection import detect_layers, stage_args  # noqa: E402


class StageArgs(unittest.TestCase):
    def test_infer_defaults_pass_only_what_is_set(self):
        self.assertEqual(stage_args.infer_args("pre", "pred", layers=["tsawa", "sabche"]),
                         ["--input", "pre", "--out", "pred", "--layers", "tsawa", "sabche", "--author-span", "last"])

    def test_infer_everything(self):
        got = stage_args.infer_args("pre", "pred", all_layers=True, device="cuda", stride=4096, dual_window=True,
                                    author_span="first", title_model="a/t", author_model="a/a")
        self.assertEqual(got, ["--input", "pre", "--out", "pred", "--all", "--device", "cuda", "--stride", "4096",
                               "--dual-window", "--author-span", "first", "--title-model", "a/t", "--author-model", "a/a"])

    def test_postprocess_defaults_and_everything(self):
        self.assertEqual(stage_args.postprocess_args("pred", "pre", "out"),
                         ["--input", "pred", "--source", "pre", "--out", "out"])
        got = stage_args.postprocess_args("pred", "pre", "out", review_threshold=0.6, title_author_review_threshold=0.4,
                                          title_author_opf=True, json=True, html=True, opf=True)
        self.assertEqual(got, ["--input", "pred", "--source", "pre", "--out", "out", "--review-threshold", "0.6",
                               "--title-author-review-threshold", "0.4", "--title-author-opf", "--json", "--html", "--opf"])

    def test_the_cli_passes_these_lists_to_the_stages(self):
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(detect_layers.preprocess, "main") as pre, \
                mock.patch.object(detect_layers.infer, "main") as inf, \
                mock.patch.object(detect_layers.postprocess, "main") as post:
            detect_layers.main(["--text", "b.txt", "--layers", "tsawa", "--json", "--out", d,
                                "--author-span", "first"])
        out = Path(d)
        pre.assert_called_once_with(["--input", "b.txt", "--out", str(out / "preprocessed")])
        inf.assert_called_once_with(["--input", str(out / "preprocessed"), "--out", str(out / "predictions"),
                                     "--layers", "tsawa", "--author-span", "first"])
        post.assert_called_once_with(["--input", str(out / "predictions"), "--source", str(out / "preprocessed"),
                                      "--out", str(out / "output"), "--json"])


if __name__ == "__main__":
    unittest.main()

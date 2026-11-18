"""--workers: books processed in several processes at once (Stage 2). Offline, no models."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tibetan_layer_detection import infer, stage_args  # noqa: E402


def square(x):
    if x == 3:
        raise ValueError("bad book")
    return x * x


class Pool(unittest.TestCase):
    def test_runs_every_job_and_one_failure_does_not_stop_the_others(self):
        # "fork" is only for this test (a tiny function, no PyTorch); the real run uses "spawn"
        got = list(infer.run_in_pool(square, [1, 2, 3, 4], workers=2, threads=None, start_method="fork"))
        ok = {job: res for job, res, err in got if err is None}
        bad = [(job, err) for job, res, err in got if err is not None]
        self.assertEqual(ok, {1: 1, 2: 4, 4: 16})
        self.assertEqual([j for j, _ in bad], [3])
        self.assertIn("bad book", str(bad[0][1]))


class Main(unittest.TestCase):
    def _folders(self, d, ids, done=()):
        pre, out = Path(d) / "pre", Path(d) / "out"
        pre.mkdir(); out.mkdir()
        for b in ids:
            (pre / f"{b}.json").write_text(json.dumps({"book_id": b, "text": "x", "char_count": 1}), encoding="utf-8")
        for b in done:
            (out / f"{b}.json").write_text("{}", encoding="utf-8")
        return pre, out

    def test_books_with_predictions_are_skipped_without_starting_workers(self):
        with tempfile.TemporaryDirectory() as d:
            pre, out = self._folders(d, ["a", "b"], done=["a", "b"])
            self.assertEqual(infer.main(["--input", str(pre), "--out", str(out), "--layers", "tsawa",
                                         "--workers", "3"]), 0)

    def test_workers_must_be_at_least_one(self):
        with tempfile.TemporaryDirectory() as d:
            pre, out = self._folders(d, ["a"])
            with self.assertRaises(SystemExit):
                infer.main(["--input", str(pre), "--out", str(out), "--layers", "tsawa", "--workers", "0"])

    def test_a_prediction_file_is_written_whole_and_leaves_no_temporary_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "b.json"
            infer._write_prediction(p, {"book_id": "b", "layers": {"tsawa": []}})
            self.assertEqual(json.loads(p.read_text(encoding="utf-8"))["book_id"], "b")
            self.assertEqual([f.name for f in Path(d).iterdir()], ["b.json"])


class ErrorSummary(unittest.TestCase):
    def test_no_errors_no_warning(self):
        self.assertIsNone(infer.error_summary({"a": {}, "b": {}}))

    def test_failed_layers_are_counted_and_named(self):
        msg = infer.error_summary({"a": {}, "b": {"tsawa": "model load failed"}, "c": {"sabche": "inference failed"}})
        self.assertIn("2 book(s)", msg)
        self.assertIn("b, c", msg)
        self.assertNotIn("--workers", msg)

    def test_out_of_memory_suggests_fewer_workers(self):
        msg = infer.error_summary({"a": {"tsawa": "inference failed: CUDA out of memory. Tried to allocate 2 GiB"}})
        self.assertIn("fewer --workers", msg)

    def test_a_long_list_of_books_is_shortened(self):
        msg = infer.error_summary({f"b{i}": {"x": "e"} for i in range(9)})
        self.assertIn("9 book(s)", msg)
        self.assertTrue(msg.split(": ", 1)[1].startswith("b0, b1, b2, b3, b4 ..."))


class Args(unittest.TestCase):
    def test_workers_are_passed_on_only_when_more_than_one(self):
        self.assertNotIn("--workers", stage_args.infer_args("p", "o", layers=["tsawa"]))
        self.assertNotIn("--workers", stage_args.infer_args("p", "o", layers=["tsawa"], workers=1))
        got = stage_args.infer_args("p", "o", layers=["tsawa"], workers=3, threads_per_worker=2)
        self.assertEqual(got[got.index("--workers"):got.index("--workers") + 2], ["--workers", "3"])
        self.assertEqual(got[got.index("--threads-per-worker"):got.index("--threads-per-worker") + 2],
                         ["--threads-per-worker", "2"])


if __name__ == "__main__":
    unittest.main()

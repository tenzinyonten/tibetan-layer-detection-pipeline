"""Tests for the title/author layer: crop, training-format sequence, decoding,
the two systems, and how Stage 3 treats the layer. Offline: a stub tokenizer, no
model download, and the skill's LLM is a stub (no paid call, no keys).

If the training dataset is on disk (TA_DATA = title_author/data/processed), one
extra test rebuilds validation sequences and checks them against the dataset's
own input_ids; it is skipped otherwise.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tibetan_layer_detection import postprocess as P  # noqa: E402
from tibetan_layer_detection import title_author as TA  # noqa: E402
from tibetan_layer_detection.layer_config import LAYERS, TITLE_AUTHOR_MODEL, TitleAuthorOptions, base_layer  # noqa: E402

CLS, SEP = 0, 1
ID2LABEL = {0: "O", 1: "B-TITLE", 2: "I-TITLE", 3: "B-AUTHOR", 4: "I-AUTHOR"}


class CharTok:
    """One token per character (id = ord + 10), real offsets, cls/sep ids 0/1."""
    cls_token_id, sep_token_id, bos_token_id, eos_token_id = CLS, SEP, None, None

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        return {"input_ids": [ord(c) + 10 for c in text], "offset_mapping": [(i, i + 1) for i in range(len(text))]}


def line_text(n_lines: int, width: int = 100) -> str:
    return "\n".join(("ཀ" * (width - 1)) for _ in range(n_lines))


def logits_for(seq, labels_at: dict[int, int], n_labels: int = 5) -> np.ndarray:
    """Logits that favour label 0 (O) everywhere except the given real-token index -> label."""
    lg = np.full((len(seq["input_ids"]), n_labels), -5.0)
    lg[:, 0] = 5.0
    for k, lab in labels_at.items():
        p = seq["real"][k]
        lg[p, :] = -5.0
        lg[p, lab] = 5.0
    return lg


class Crop(unittest.TestCase):
    def test_short_work_is_whole(self):
        self.assertEqual(TA.crop_regions("ཀ" * 6000), [("whole", 0, 6000)])

    def test_long_work_snaps_to_line_starts(self):
        text = line_text(100, 100)                     # 9,999 characters, line starts every 100
        (hn, hs, he), (tn, ts, te) = TA.crop_regions(text)
        self.assertEqual((hn, hs, tn, te), ("head", 0, "tail", len(text)))
        self.assertEqual(he, 3100)                     # the first line start after position 3,000 (as in training)
        self.assertEqual(text[he - 1], "\n")
        self.assertLessEqual(abs(ts - (len(text) - 3000)), 1)   # 6,999 is a newline, so the tail starts on the next line, at 7,000
        self.assertTrue(ts == 0 or text[ts - 1] == "\n")

    def test_head_and_tail_that_meet_become_whole(self):
        text = "ཀ" * 6500                              # no line break or marks: the edges do not move, 3000 < 3500
        self.assertEqual(len(TA.crop_regions(text)), 2)
        text = "ཀ" * 3200 + "་" + "ཀ" * 3200            # snap_forward moves past 3,200 to the tsek, so head end > tail start
        self.assertEqual(TA.crop_regions(text, 3000)[0][0], "whole")


class Sequence(unittest.TestCase):
    def test_short_work_has_one_sep(self):
        seq = TA.build_sequence("ཀཁག", CharTok())
        self.assertEqual(seq["input_ids"][0], CLS)
        self.assertEqual(seq["input_ids"].count(SEP), 1)
        self.assertEqual(seq["input_ids"][-1], SEP)
        self.assertEqual(seq["real"], [1, 2, 3])
        self.assertEqual(seq["offsets"], [(0, 1), (1, 2), (2, 3)])

    def test_long_work_is_head_sep_tail_sep_with_offsets_in_the_full_text(self):
        text = line_text(100, 100)
        seq = TA.build_sequence(text, CharTok())
        (_, _, he), (_, ts, te) = seq["regions"]
        ids = seq["input_ids"]
        self.assertEqual(ids[0], CLS)
        self.assertEqual(ids.count(SEP), 2)
        self.assertEqual(ids[1 + he], SEP)             # the separator sits right after the head tokens
        self.assertEqual(len(seq["real"]), he + (te - ts))
        self.assertEqual(seq["offsets"][he], (ts, ts + 1))   # first tail token -> its place in the full text
        self.assertEqual(seq["offsets"][-1], (te - 1, te))
        for pos in seq["real"]:
            self.assertNotIn(ids[pos], (CLS, SEP))


class Viterbi(unittest.TestCase):
    def test_transition_matrix_matches_the_reference_scorer(self):
        # the matrix in title_author/src/ta_mmbert_score.py, rebuilt here
        NEG, PEN = -1e9, 12.0
        ref = np.zeros((5, 5))
        for i in range(5):
            for j in range(5):
                if j == 2 and i not in (1, 2): ref[i, j] = NEG; continue
                if j == 4 and i not in (3, 4): ref[i, j] = NEG; continue
                cont = 2 if i in (1, 2) else 4 if i in (3, 4) else None
                if i != 0 and j != cont: ref[i, j] -= PEN
        np.testing.assert_array_equal(TA.transition_matrix(ID2LABEL, PEN), ref)

    def test_a_break_costs_the_penalty(self):
        lg = np.zeros((5, 5))
        lg[:, 0] = 1.0
        lg[0, 1] = 3.0; lg[1, 0] = 0.5; lg[1, 2] = 0.4; lg[2, 2] = 3.0       # B I I ... one span is cheaper than O inside
        lg[1, 2] = 2.9
        path = TA.viterbi(lg, ID2LABEL, 12.0)
        self.assertEqual(path[:3].tolist(), [1, 2, 2])

    def test_i_cannot_open_a_sequence(self):
        lg = np.zeros((3, 5)); lg[:, 2] = 9.0
        self.assertNotEqual(int(TA.viterbi(lg, ID2LABEL, 12.0)[0]), 2)


class Decode(unittest.TestCase):
    def setUp(self):
        self.text = line_text(100, 100)
        self.seq = TA.build_sequence(self.text, CharTok())
        self.he = self.seq["regions"][0][2]

    def test_title_first_author_last_and_blocks(self):
        # title at the first two head tokens, a second title later, author twice in the tail
        marks = {0: 1, 1: 2, 20: 1, 21: 2,
                 self.he + 5: 3, self.he + 6: 4, self.he + 50: 3, self.he + 51: 4}
        out = TA.decode_fields(self.text, self.seq, logits_for(self.seq, marks), ID2LABEL, ["title", "author"])
        t, a = out["title"][0], out["author"][0]
        self.assertEqual((t["start"], t["end"], t["block"]), (0, 2, "head"))
        ts = self.seq["regions"][1][1]
        self.assertEqual((a["start"], a["end"], a["block"]), (ts + 50, ts + 52, "tail"))
        self.assertEqual(t["system"], "model")
        self.assertGreater(t["confidence"], 0.99)

    def test_nothing_found_gives_empty_lists(self):
        out = TA.decode_fields(self.text, self.seq, logits_for(self.seq, {}), ID2LABEL, ["title", "author"])
        self.assertEqual(out, {"title": [], "author": []})

    def test_special_tokens_are_ignored(self):
        lg = logits_for(self.seq, {0: 1, 1: 2})
        for pos, tid in enumerate(self.seq["input_ids"]):
            if tid in (CLS, SEP):
                lg[pos, :] = -5.0; lg[pos, 3] = 50.0        # screams B-AUTHOR at every special token
        out = TA.decode_fields(self.text, self.seq, lg, ID2LABEL, ["title", "author"])
        self.assertEqual(out["author"], [])
        self.assertEqual(len(out["title"]), 1)

    def test_a_span_starts_at_the_letter_not_the_space(self):
        text = "ཀཁ གངཅ"
        seq = TA.build_sequence(text, CharTok())
        marks = {2: 1, 3: 2, 4: 2}                          # B on the space token, then the letters
        out = TA.decode_fields(text, seq, logits_for(seq, marks), ID2LABEL, ["title"])
        self.assertEqual(text[out["title"][0]["start"]:out["title"][0]["end"]], "གངཅ")

    def test_whole_work_block(self):
        text = "ཀཁགང"
        seq = TA.build_sequence(text, CharTok())
        out = TA.decode_fields(text, seq, logits_for(seq, {0: 3, 1: 4}), ID2LABEL, ["author"])
        self.assertEqual(out["author"][0]["block"], "whole")

    def test_single_entity_models(self):
        named = {0: "O", 1: "B-AUTHOR", 2: "I-AUTHOR"}
        bare = {0: "O", 1: "B", 2: "I"}
        self.assertEqual(TA.entity_for("author", named), (True, "AUTHOR"))
        self.assertEqual(TA.entity_for("title", named), (False, None))
        self.assertEqual(TA.entity_for("title", bare), (True, None))
        text = "ཀཁགངཅཆ"                                    # the span is in the middle: one that runs to the end of the sequence pays no break penalty
        seq = TA.build_sequence(text, CharTok())
        for id2label in (named, bare):
            lg = logits_for(seq, {1: 1, 2: 2}, n_labels=3)
            out = TA.decode_fields(text, seq, lg, id2label, ["author"])
            self.assertEqual((out["author"][0]["start"], out["author"][0]["end"]), (1, 3))


class Systems(unittest.TestCase):
    def setUp(self):
        self._rm, self._rs = TA.run_model, TA.run_skill

    def tearDown(self):
        TA.run_model, TA.run_skill = self._rm, self._rs

    def span(self, s, e, **kw):
        return {"start": s, "end": e, "confidence": 0.9, "window_votes": 1, **kw}

    def test_both_puts_the_answers_side_by_side_with_an_agreement_flag(self):
        TA.run_model = lambda text, fields, repo, device, k=3000, bp=12.0: {
            "title": [self.span(0, 10, system="model")], "author": [self.span(50, 60, system="model")]}
        TA.run_skill = lambda text, fields, opts, call_llm=None, k=3000: {
            "title": [self.span(0, 12, system="skill", confidence=None)], "author": [self.span(200, 210, system="skill", confidence=None)]}
        layers, _, errors = TA.run_title_author("x" * 300, ["title", "author"], TitleAuthorOptions(system="both"), "cpu")
        self.assertEqual(set(layers), {"title", "author", "title_skill", "author_skill"})
        self.assertEqual(errors, {})
        self.assertTrue(layers["title"][0]["systems_agree"])        # IoU 10/12
        self.assertTrue(layers["title_skill"][0]["systems_agree"])
        self.assertFalse(layers["author"][0]["systems_agree"])      # disjoint
        self.assertEqual(layers["author_skill"][0]["system"], "skill")

    def test_model_mode_never_touches_the_skill(self):
        def boom(*a, **k):
            raise AssertionError("the skill must not be used in model mode")
        TA.run_skill = boom
        TA.run_model = lambda text, fields, repo, device, k=3000, bp=12.0: {f: [] for f in fields}
        layers, _, errors = TA.run_title_author("x", ["title", "author"], TitleAuthorOptions(), "cpu")
        self.assertEqual((layers, errors), ({"title": [], "author": []}, {}))

    def test_skill_mode_uses_the_plain_names(self):
        TA.run_skill = lambda text, fields, opts, call_llm=None, k=3000: {f: [self.span(0, 3, system="skill")] for f in fields}
        layers, _, _ = TA.run_title_author("x" * 10, ["title", "author"], TitleAuthorOptions(system="skill"), "cpu")
        self.assertEqual(set(layers), {"title", "author"})

    def test_two_models_run_twice_one_model_runs_once(self):
        calls = []
        def fake(text, fields, repo, device, k=3000, bp=12.0):
            calls.append((repo, tuple(fields)))
            return {f: [] for f in fields}
        TA.run_model = fake
        TA.run_title_author("x", ["title", "author"], TitleAuthorOptions(), "cpu")
        self.assertEqual(calls, [(TITLE_AUTHOR_MODEL, ("title", "author"))])
        calls.clear()
        TA.run_title_author("x", ["title", "author"], TitleAuthorOptions(title_model="a/t", author_model="a/a"), "cpu")
        self.assertEqual(calls, [("a/t", ("title",)), ("a/a", ("author",))])

    def test_model_failure_is_an_error_not_a_crash(self):
        def fail(*a, **k):
            raise OSError("no access")
        TA.run_model = fail
        layers, _, errors = TA.run_title_author("x", ["title", "author"], TitleAuthorOptions(), "cpu")
        self.assertEqual(layers, {})
        self.assertIn("no access", errors["title"])
        self.assertIn("no access", errors["author"])

    @unittest.skipUnless(os.environ.get("TA_SKILL_DIR") and (Path(os.environ["TA_SKILL_DIR"]) / "title_author_skill.py").is_file(),
                         "the skill is not bundled here: set TA_SKILL_DIR to its scripts folder")
    def test_skill_locates_its_answers_with_a_stub_llm(self):
        text = "༄༅། །བཀྲ་ཤིས་འཁྲུལ་པ་དཀར་པོ། ། རྒྱལ་བ་ཐམས་ཅད། །\n" * 3 + "རྩོམ་པ་པོ། དགེ་སློང་བློ་བཟང་གིས་སྦྱར།"
        stub = lambda system, user: json.dumps(  # noqa: E731
            {"title": "བཀྲ་ཤིས་འཁྲུལ་པ་དཀར་པོ", "title_region": "head",
             "author": "དགེ་སློང་བློ་བཟང་", "author_region": "head"}, ensure_ascii=False)
        out = TA.run_skill(text, ["title", "author"], TitleAuthorOptions(system="skill"), call_llm=stub)
        t, a = out["title"][0], out["author"][0]
        self.assertEqual(text[t["start"]:t["end"]], "བཀྲ་ཤིས་འཁྲུལ་པ་དཀར་པོ")
        self.assertEqual(text[a["start"]:a["end"]], "དགེ་སློང་བློ་བཟང་")
        self.assertEqual((t["system"], t["confidence"], t["block"]), ("skill", None, "whole"))
        self.assertIn("review_needed", t)


class Config(unittest.TestCase):
    def test_layers(self):
        self.assertEqual((LAYERS["title"].kind, LAYERS["author"].kind), ("head_tail", "head_tail"))
        for name in ("tsawa", "sabche", "chapter", "quotation", "yigchung"):
            self.assertEqual(LAYERS[name].kind, "window")
        self.assertEqual(LAYERS["title"].break_penalty, 12.0)
        self.assertEqual(base_layer("author_skill"), "author")
        self.assertEqual(base_layer("tsawa"), "tsawa")

    def test_defaults_are_free_and_local(self):
        o = TitleAuthorOptions()
        self.assertEqual((o.system, o.title_model, o.author_model, o.opf), ("model", TITLE_AUTHOR_MODEL, TITLE_AUTHOR_MODEL, False))


class Stage3(unittest.TestCase):
    def test_short_author_is_not_dropped_but_other_short_spans_are(self):
        sp = lambda n: [{"start": 0, "end": n, "confidence": 0.9}]  # noqa: E731
        kept, dropped = P.drop_short_spans({"author": sp(3), "author_skill": sp(3), "sabche": sp(3)})
        self.assertEqual(len(kept["author"]), 1)
        self.assertEqual(len(kept["author_skill"]), 1)
        self.assertEqual(kept["sabche"], [])
        self.assertEqual(dropped, {"author": 0, "author_skill": 0, "sabche": 1})

    def test_threshold_precedence(self):
        self.assertEqual(P.threshold_for("title", 0.7), 0.7)                # the one global threshold
        self.assertEqual(P.threshold_for("title", 0.6), 0.6)
        self.assertEqual(P.threshold_for("title", 0.6, 0.4), 0.4)
        self.assertEqual(P.threshold_for("author_skill", None, 0.4), 0.4)
        self.assertEqual(P.threshold_for("tsawa", 0.6, 0.4), 0.6)          # the title/author flag leaves other layers alone

    def test_review_flags(self):
        low = {"confidence": 0.3}
        self.assertTrue(P.needs_review(low, "author", 0.7))
        self.assertFalse(P.needs_review({"confidence": 0.95}, "author", 0.7))
        self.assertTrue(P.needs_review({"confidence": None, "review_needed": True}, "author_skill", 0.7))
        self.assertFalse(P.needs_review({"confidence": None, "review_needed": False}, "author_skill", 0.7))

    def test_payload_keeps_the_title_author_fields(self):
        s = {"start": 1, "end": 9, "confidence": None, "block": "tail", "system": "skill",
             "review_reasons": ["author was weak"], "systems_agree": False, "review_needed": True}
        out = P._span_payload(s, "author_skill", 0.7)
        self.assertEqual((out["confidence"], out["block"], out["system"], out["review_reasons"], out["systems_agree"]),
                         (None, "tail", "skill", ["author was weak"], False))
        self.assertTrue(out["review_needed"])

    def test_opf_for_title_author_is_a_clear_error(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(SystemExit) as cm:
                P.main(["--input", d, "--source", d, "--out", d, "--title-author-opf"])
        self.assertIn("not implemented", str(cm.exception))

    def test_opf_output_skips_title_author(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "pre").mkdir(); (d / "pred").mkdir()
            txt = d / "b.txt"
            body = "བཀྲ་ཤིས་བདེ་ལེགས། " * 20
            txt.write_text(body, encoding="utf-8")
            (d / "pre" / "b.json").write_text(json.dumps(
                {"book_id": "b", "text": body, "char_count": len(body), "source_path": str(txt)}), encoding="utf-8")
            span = lambda s, e: {"start": s, "end": e, "confidence": 0.9, "window_votes": 1}  # noqa: E731
            (d / "pred" / "b.json").write_text(json.dumps({"book_id": "b", "layers": {
                "sabche": [span(0, 40)], "title": [span(0, 9)], "author": [span(50, 58)]}}), encoding="utf-8")
            P.main(["--input", str(d / "pred"), "--source", str(d / "pre"), "--out", str(d / "out"), "--json", "--opf"])
            names = sorted(p.name for p in (d / "out").rglob("*.yml"))
            self.assertEqual(names, ["Sabche.yml"])
            res = json.loads((d / "out" / "b.json").read_text(encoding="utf-8"))
            self.assertEqual({k: len(v) for k, v in res["layers"].items()}, {"sabche": 1, "title": 1, "author": 1})


TA_DATA = Path(os.environ.get("TA_DATA", ROOT / "title_author/data/processed"))  # not in this repo: set TA_DATA


@unittest.skipUnless((TA_DATA / "gold_dataset_all_strat_single_v3_both").is_dir()
                     and (TA_DATA / "parquet_export_strat_both/validation.parquet").is_file(),
                     "training dataset not on disk (set TA_DATA)")
class MatchesTheTrainingData(unittest.TestCase):
    def test_validation_sequences_equal_the_dataset_input_ids(self):
        import pandas as pd
        from datasets import load_from_disk
        from transformers import AutoTokenizer
        ds = load_from_disk(str(TA_DATA / "gold_dataset_all_strat_single_v3_both"))["validation"]
        pq = pd.read_parquet(TA_DATA / "parquet_export_strat_both/validation.parquet").set_index("work_id")
        tok = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-base")
        tok.model_max_length = int(1e12)
        keep = TA.crop_regions
        try:
            for i in range(0, len(ds), 25):                    # every 25th validation row
                row = ds[i]
                r = pq.loc[row["work_id"]]
                head, tail = r.text_head, (r.text_tail or "")
                if str(r.whole_work).lower() == "true" or not tail:
                    text, regs = head, [("whole", 0, len(head))]
                else:
                    text = head + tail
                    regs = [("head", 0, len(head)), ("tail", len(head), len(text))]
                TA.crop_regions = lambda t, k=3000, _r=regs: _r  # the cropped texts are what the dataset holds
                seq = TA.build_sequence(text, tok)
                self.assertEqual(seq["input_ids"], row["input_ids"][:sum(row["attention_mask"])], row["work_id"])
        finally:
            TA.crop_regions = keep


if __name__ == "__main__":
    unittest.main()

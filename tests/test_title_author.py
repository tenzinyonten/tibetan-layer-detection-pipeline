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
from tibetan_layer_detection.layer_config import (AUTHOR_MODEL, JOINT_MODEL, LAYERS, TITLE_MODEL, TitleAuthorOptions,  # noqa: E402
                                 base_layer)

SKILL_DIR = Path(os.environ.get("TA_SKILL_DIR") or ROOT / "title_author/skills/tibetan-title-author/scripts")
needs_skill = unittest.skipUnless((SKILL_DIR / "title_author_skill.py").is_file(),
                                  "the skill is not here: set TA_SKILL_DIR to its scripts folder")

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


class StripUnderscore(unittest.TestCase):
    def test_tokens_are_built_without_underscores_but_offsets_point_into_the_original(self):
        text = "ཀ_ཁ_ག ང"
        plain = TA.build_sequence(text, CharTok(), strip_underscore=False)
        strip = TA.build_sequence(text, CharTok(), strip_underscore=True)
        self.assertEqual(len(plain["real"]), 7)
        self.assertEqual(len(strip["real"]), 5)
        self.assertNotIn(ord("_") + 10, strip["input_ids"])
        self.assertEqual(strip["offsets"], [(0, 1), (2, 3), (4, 5), (5, 6), (6, 7)])   # ཀ ཁ ག space ང, in the original text

    def test_a_decoded_span_may_contain_an_underscore_and_is_an_exact_substring(self):
        text = "ཀ_ཁ_གངཅ"
        seq = TA.build_sequence(text, CharTok(), strip_underscore=True)
        out = TA.decode_fields(text, seq, logits_for(seq, {0: 1, 1: 2, 2: 2}, n_labels=3), {0: "O", 1: "B-TITLE", 2: "I-TITLE"}, ["title"])
        sp = out["title"][0]
        self.assertEqual(text[sp["start"]:sp["end"]], "ཀ_ཁ_ག")

    def test_text_without_underscores_is_unchanged(self):
        self.assertEqual(TA.build_sequence("ཀཁག", CharTok(), strip_underscore=True), TA.build_sequence("ཀཁག", CharTok()))

    def test_which_models_get_stripped_text(self):
        self.assertTrue(TA.uses_stripped_text({0: "O", 1: "B-AUTHOR", 2: "I-AUTHOR"}))
        self.assertTrue(TA.uses_stripped_text({0: "O", 1: "B", 2: "I"}))
        self.assertFalse(TA.uses_stripped_text(ID2LABEL))


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

    def test_single_entity_matrix_matches_the_reference_scorer(self):
        # ta_mmbert_single_score.py trans_for(), rebuilt here, for a 3-label model
        NEG, PEN = -1e9, 6.0
        ref = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                if j == 2 and i not in (1, 2): ref[i, j] = NEG; continue
                if i != 0 and j != (2 if i in (1, 2) else None): ref[i, j] -= PEN
        for labels in ({0: "O", 1: "B-AUTHOR", 2: "I-AUTHOR"}, {0: "O", 1: "B", 2: "I"}):
            np.testing.assert_array_equal(TA.transition_matrix(labels, PEN), ref)

    def test_break_penalty_depends_on_the_model(self):
        self.assertEqual(TA.default_break_penalty({0: "O", 1: "B-TITLE", 2: "I-TITLE"}), 6.0)
        self.assertEqual(TA.default_break_penalty({0: "O", 1: "B", 2: "I"}), 6.0)
        self.assertEqual(TA.default_break_penalty(ID2LABEL), 12.0)

    def test_the_penalty_decides_whether_a_gap_splits_a_span(self):
        # B I [gap] B I O: splitting beats bridging the gap by 5.5 minus the penalty, so a small penalty
        # splits and a large one bridges (the bridge makes tokens 2 and 3 I, which is legal after an I)
        labels = {0: "O", 1: "B-AUTHOR", 2: "I-AUTHOR"}
        lg = np.zeros((6, 3))
        lg[0, 1] = 5; lg[1, 2] = 5; lg[2, 0] = 1.5; lg[3, 1] = 5; lg[3, 2] = 1; lg[4, 2] = 5; lg[5, 0] = 5
        spans = lambda pen: TA.entity_spans(TA.viterbi(lg, labels, pen), labels, "AUTHOR")  # noqa: E731
        self.assertEqual(spans(0.5), [(0, 1), (3, 4)])
        self.assertEqual(len(spans(20.0)), 1)               # one bridged span (a large penalty even keeps it running to the last token)
        self.assertEqual(spans(20.0)[0][0], 0)

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
        TA.run_model = lambda text, fields, repo, device, k=3000, bp=None, strip=None: {
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
        TA.run_model = lambda text, fields, repo, device, k=3000, bp=None, strip=None: {f: [] for f in fields}
        layers, _, errors = TA.run_title_author("x", ["title", "author"], TitleAuthorOptions(), "cpu")
        self.assertEqual((layers, errors), ({"title": [], "author": []}, {}))

    def test_skill_mode_uses_the_plain_names(self):
        TA.run_skill = lambda text, fields, opts, call_llm=None, k=3000: {f: [self.span(0, 3, system="skill")] for f in fields}
        layers, _, _ = TA.run_title_author("x" * 10, ["title", "author"], TitleAuthorOptions(system="skill"), "cpu")
        self.assertEqual(set(layers), {"title", "author"})

    def test_two_models_run_twice_one_model_runs_once(self):
        calls = []
        def fake(text, fields, repo, device, k=3000, bp=None, strip=None):
            calls.append((repo, tuple(fields)))
            return {f: [] for f in fields}
        TA.run_model = fake
        TA.run_title_author("x", ["title", "author"], TitleAuthorOptions(), "cpu")      # the defaults: two single-entity models
        self.assertEqual(calls, [(TITLE_MODEL, ("title",)), (AUTHOR_MODEL, ("author",))])
        calls.clear()
        TA.run_title_author("x", ["title", "author"], TitleAuthorOptions(title_model=JOINT_MODEL, author_model=JOINT_MODEL), "cpu")
        self.assertEqual(calls, [(JOINT_MODEL, ("title", "author"))])                   # the joint model: one forward pass
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

    @needs_skill
    def test_skill_locates_its_answers_with_a_stub_llm(self):
        text = "༄༅། །བཀྲ་ཤིས་འཁྲུལ་པ་དཀར་པོ། ། རྒྱལ་བ་ཐམས་ཅད། །\n" * 3 + "རྩོམ་པ་པོ། དགེ་སློང་བློ་བཟང་གིས་སྦྱར།"
        stub = lambda system, user: json.dumps(  # noqa: E731
            {"title": "བཀྲ་ཤིས་འཁྲུལ་པ་དཀར་པོ", "title_region": "head",
             "author": "དགེ་སློང་བློ་བཟང་", "author_region": "head"}, ensure_ascii=False)
        out = TA.run_skill(text, ["title", "author"], TitleAuthorOptions(system="skill", skill_dir=str(SKILL_DIR)), call_llm=stub)
        t, a = out["title"][0], out["author"][0]
        self.assertEqual(text[t["start"]:t["end"]], "བཀྲ་ཤིས་འཁྲུལ་པ་དཀར་པོ")
        self.assertEqual(text[a["start"]:a["end"]], "དགེ་སློང་བློ་བཟང་")
        self.assertEqual((t["system"], t["confidence"], t["block"]), ("skill", None, "whole"))
        self.assertIn("review_needed", t)


class PerFieldSystems(unittest.TestCase):
    """--title-system / --author-system: a fixed choice per field, never a rule per work."""

    def setUp(self):
        self._rm, self._rs = TA.run_model, TA.run_skill
        self.calls = []
        span = lambda s, e, **kw: {"start": s, "end": e, "confidence": 0.9, "window_votes": 1, **kw}  # noqa: E731

        def fake_model(text, fields, repo, device, k=3000, bp=None, strip=None):
            self.calls.append(("model", repo, tuple(fields)))
            return {f: [span(0, 5, system="model")] for f in fields}

        def fake_skill(text, fields, opts, call_llm=None, k=3000):
            self.calls.append(("skill", tuple(fields)))
            return {f: [span(10, 20, system="skill", confidence=None, review_needed=True,
                             review_reasons=["author was weak"])] for f in fields}
        TA.run_model, TA.run_skill = fake_model, fake_skill

    def tearDown(self):
        TA.run_model, TA.run_skill = self._rm, self._rs

    def test_title_from_the_model_and_author_from_the_skill(self):
        opts = TitleAuthorOptions(title_system="model", author_system="skill")
        layers, _, errors = TA.run_title_author("x" * 50, ["title", "author"], opts, "cpu")
        self.assertEqual(self.calls, [("model", TITLE_MODEL, ("title",)), ("skill", ("author",))])   # the model is not asked for the author
        self.assertEqual(set(layers), {"title", "author"})                                          # plain names, no _skill keys
        self.assertEqual((layers["title"][0]["system"], layers["author"][0]["system"]), ("model", "skill"))
        self.assertEqual(layers["author"][0]["review_reasons"], ["author was weak"])                 # the skill's review flag carries over
        self.assertTrue(layers["author"][0]["review_needed"])
        self.assertNotIn("systems_agree", layers["author"][0])                                       # no comparison: one system per field
        self.assertEqual(errors, {})

    def test_defaults_are_the_model_for_both_fields_and_no_skill_call(self):
        layers, _, _ = TA.run_title_author("x" * 50, ["title", "author"], TitleAuthorOptions(), "cpu")
        self.assertEqual([c[0] for c in self.calls], ["model", "model"])
        self.assertEqual({f: layers[f][0]["system"] for f in layers}, {"title": "model", "author": "model"})

    def test_the_shortcut_still_sets_both_fields(self):
        TA.run_title_author("x" * 50, ["title", "author"], TitleAuthorOptions(system="skill"), "cpu")
        self.assertEqual(self.calls, [("skill", ("title", "author"))])
        self.assertEqual(TitleAuthorOptions(system="skill", author_system="model").field_system("author"), "model")  # a per-field value wins

    def test_a_skill_failure_is_an_error_for_that_field_with_no_fallback_to_the_model(self):
        def boom(text, fields, opts, call_llm=None, k=3000):
            raise RuntimeError("set GEMINI_API_KEY in the environment")
        TA.run_skill = boom
        layers, _, errors = TA.run_title_author("x" * 50, ["title", "author"],
                                                TitleAuthorOptions(title_system="model", author_system="skill"), "cpu")
        self.assertEqual(set(layers), {"title"})
        self.assertEqual(set(errors), {"author"})
        self.assertIn("GEMINI_API_KEY", errors["author"])
        self.assertEqual([c[0] for c in self.calls], ["model"])                                      # the model was NOT asked for the author

    def test_only_the_requested_fields_run(self):
        opts = TitleAuthorOptions(title_system="model", author_system="skill")
        layers, _, _ = TA.run_title_author("x" * 50, ["author"], opts, "cpu")
        self.assertEqual(self.calls, [("skill", ("author",))])
        self.assertEqual(set(layers), {"author"})

    def test_invalid_choices_are_errors(self):
        with self.assertRaises(ValueError):
            TitleAuthorOptions(system="both", author_system="skill")        # both = side by side; cannot mix with a per-field choice
        with self.assertRaises(ValueError):
            TitleAuthorOptions(title_system="both")
        with self.assertRaises(ValueError):
            TitleAuthorOptions(system="nope")

    def test_the_cost_note(self):
        fields = ["title", "author"]
        self.assertIsNone(TA.skill_note(TitleAuthorOptions(), fields, 100))                         # the free default prints nothing
        note = TA.skill_note(TitleAuthorOptions(title_system="model", author_system="skill"), fields, 100)
        self.assertIn("PAID", note)
        self.assertIn("Rs 29.00", note)                                                             # 100 books x about 0.29
        self.assertIn("author from the skill", note)
        self.assertIsNone(TA.skill_note(TitleAuthorOptions(author_system="skill"), ["title"], 100))  # the skill is not used for a title-only run
        self.assertIn("both systems", TA.skill_note(TitleAuthorOptions(system="both"), fields, 1))


@needs_skill
class SkillStubOneCall(unittest.TestCase):
    def test_one_call_per_work_no_second_look_and_the_skills_title_is_ignored(self):
        text = "༄༅། །བཀྲ་ཤིས་འཁྲུལ་པ་དཀར་པོ། ། རྒྱལ་བ་ཐམས་ཅད། །\n" * 3 + "རྩོམ་པ་པོ། དགེ་སློང་བློ་བཟང་གིས་སྦྱར།"
        n = []

        def stub(system, user):
            n.append(1)
            return json.dumps({"title": "བཀྲ་ཤིས་འཁྲུལ་པ་དཀར་པོ", "title_region": "head",
                               "author": "དགེ་སློང་བློ་བཟང་", "author_region": "head"}, ensure_ascii=False)
        opts = TitleAuthorOptions(title_system="model", author_system="skill", skill_dir=str(SKILL_DIR))
        out = TA.run_skill(text, ["author"], opts, call_llm=stub)
        self.assertEqual(len(n), 1)                                   # one LLM call, no second look
        self.assertEqual(set(out), {"author"})                        # its title answer was ignored
        a = out["author"][0]
        self.assertEqual(text[a["start"]:a["end"]], "དགེ་སློང་བློ་བཟང་")
        self.assertIn("review_reasons", a)


class Config(unittest.TestCase):
    def test_layers(self):
        self.assertEqual((LAYERS["title"].kind, LAYERS["author"].kind), ("head_tail", "head_tail"))
        for name in ("tsawa", "sabche", "chapter", "quotation", "yigchung"):
            self.assertEqual(LAYERS[name].kind, "window")
        self.assertEqual((LAYERS["title"].break_penalty, LAYERS["title"].repo), (6.0, TITLE_MODEL))
        self.assertEqual((LAYERS["author"].break_penalty, LAYERS["author"].repo), (6.0, AUTHOR_MODEL))
        self.assertEqual(base_layer("author_skill"), "author")
        self.assertEqual(base_layer("tsawa"), "tsawa")

    def test_defaults_are_free_and_local(self):
        o = TitleAuthorOptions()
        self.assertEqual((o.system, o.title_model, o.author_model, o.opf), ("model", TITLE_MODEL, AUTHOR_MODEL, False))
        self.assertEqual((TITLE_MODEL, AUTHOR_MODEL), ("Yontenn/mmbert-title-only", "Yontenn/mmbert-author-only"))
        self.assertIsNone(o.break_penalty)                # per model: 6 single-entity, 12 joint
        self.assertIsNone(o.strip_underscore)             # per model: single-entity models are trained on text without "_"


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

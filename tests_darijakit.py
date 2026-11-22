"""Unit tests for DarijaKit. Stdlib only: python -m unittest tests_darijakit"""

import json
import os
import tempfile
import unittest

import darijakit as dk


class TestDetectScript(unittest.TestCase):
    def test_arabic(self):
        self.assertEqual(dk.detect_script("صافي غادي نمشي دابا"), "arabic")

    def test_arabizi_with_digits(self):
        self.assertEqual(dk.detect_script("ch7al mn sa3a"), "arabizi")
        self.assertEqual(dk.detect_script("bghit nakol"), "arabizi")  # word map
        self.assertEqual(dk.detect_script("zwerty plok"), "other")
        self.assertEqual(dk.detect_script("3afak a sahbi"), "arabizi")

    def test_arabizi_known_word(self):
        self.assertEqual(dk.detect_script("salam labas"), "arabizi")

    def test_other_latin(self):
        self.assertEqual(dk.detect_script("Bonjour, comment allez-vous"), "other")
        self.assertEqual(dk.detect_script("The quick brown fox"), "other")

    def test_mixed(self):
        self.assertEqual(dk.detect_script("salam خويا"), "mixed")


class TestArabizi(unittest.TestCase):
    def test_word_map(self):
        self.assertEqual(dk.arabizi_to_arabic("ch7al"), "شحال")
        self.assertEqual(dk.arabizi_to_arabic("bghit"), "بغيت")
        self.assertEqual(dk.arabizi_to_arabic("wakha"), "واخا")

    def test_character_fallback(self):
        # Unknown word falls back to char-level rules.
        self.assertEqual(dk.arabizi_to_arabic("3chir"), "عشير")

    def test_sentence(self):
        out = dk.arabizi_to_arabic("ch7al mn sa3a khasni nkhdem?")
        self.assertIn("شحال", out)
        self.assertTrue(out.endswith("?"))

    def test_passthrough_numbers(self):
        self.assertEqual(dk.arabizi_to_arabic("100"), "100")


class TestNormalize(unittest.TestCase):
    def test_alef_unification(self):
        self.assertEqual(dk.normalize_arabic("أكل إياس آمن"), "اكل اياس امن")

    def test_diacritics_and_tatweel(self):
        self.assertEqual(dk.normalize_arabic("كِتَاب"), "كتاب")
        self.assertEqual(dk.normalize_arabic("عـربي"), "عربي")

    def test_ya_unification(self):
        self.assertEqual(dk.normalize_arabic("فتى"), "فتي")

    def test_darija_variants(self):
        self.assertEqual(dk.normalize_darija("بش نمشي"), "باش نمشي")
        self.assertEqual(dk.normalize_darija("داب صافي"), "دابا صافي")

    def test_smart_normalize_arabizi(self):
        self.assertEqual(dk.normalize("bghit"), "بغيت")

    def test_whitespace(self):
        self.assertEqual(dk.normalize_arabic("  سلام   خويا  "), "سلام خويا")


class TestEvalScoring(unittest.TestCase):
    def _item(self, kind, expected):
        return dk.EvalItem(id="t", kind=kind, prompt="p", expected=expected)

    def test_mcq_pass(self):
        ok, _ = dk.score_item(self._item("mcq", ["أ"]), "الجواب الصحيح هو أ")
        self.assertTrue(ok)

    def test_mcq_latin_letter(self):
        ok, _ = dk.score_item(self._item("mcq", ["أ", "a"]), "Answer: A")
        self.assertTrue(ok)

    def test_mcq_fail(self):
        ok, _ = dk.score_item(self._item("mcq", ["ج"]), "الجواب هو أ بالتأكيد")
        self.assertFalse(ok)

    def test_translation_pass(self):
        item = self._item("translation", ["train station"])
        ok, _ = dk.score_item(item, "Where is the train station?")
        self.assertTrue(ok)

    def test_translation_normalized(self):
        # Alef variants and case should not matter.
        item = self._item("translation", ["أريد أن أشتري"])
        ok, _ = dk.score_item(item, "اريد ان اشتري الخضرة")
        self.assertTrue(ok)

    def test_exact(self):
        item = self._item("exact", ["فاطمة"])
        ok, _ = dk.score_item(item, "فاطمة")
        self.assertTrue(ok)
        ok, _ = dk.score_item(item, "فاطمة الزهراء")
        self.assertFalse(ok)


class TestEvalRunner(unittest.TestCase):
    def test_run_eval_with_fake_client(self):
        items = [
            dk.EvalItem(id="1", kind="mcq", prompt="p", expected=["أ"]),
            dk.EvalItem(id="2", kind="contains", prompt="p", expected=["x"]),
        ]
        report = dk.run_eval(
            items,
            model="fake",
            client=lambda prompt, system=None: "أ" if prompt == "p" else "",
        )
        self.assertEqual(report.total, 2)
        self.assertEqual(report.passed, 1)
        cats = report.by_category()
        self.assertEqual(cats["general"], (1, 2))

    def test_run_eval_survives_client_errors(self):
        items = [dk.EvalItem(id="1", kind="mcq", prompt="p", expected=["أ"])]

        def boom(prompt, system=None):
            raise ConnectionError("endpoint down")

        report = dk.run_eval(items, model="fake", client=boom)
        self.assertEqual(report.passed, 0)
        self.assertIn("error", report.results[0].score_detail)

    def test_reports_render(self):
        items = [dk.EvalItem(id="1", kind="mcq", prompt="p", expected=["أ"])]
        report = dk.run_eval(items, model="fake",
                             client=lambda p, s=None: "أ")
        md = dk.report_to_markdown(report)
        self.assertIn("1/1", md)
        payload = json.loads(dk.report_to_json(report))
        self.assertEqual(payload["passed"], 1)


class TestEvalset(unittest.TestCase):
    def test_bundled_evalset_loads(self):
        path = os.path.join(os.path.dirname(__file__), "evalset.jsonl")
        items = dk.load_evalset(path)
        self.assertGreaterEqual(len(items), 25)
        for item in items:
            self.assertIn(item.kind, {"mcq", "translation", "contains", "exact"})
            self.assertTrue(item.expected)

    def test_load_evalset_reports_line(self):
        with tempfile.NamedTemporaryFile(
                "w", suffix=".jsonl", delete=False, encoding="utf-8") as fh:
            fh.write('{"id": "x", "kind": "mcq"}\n')
            path = fh.name
        try:
            with self.assertRaises(ValueError) as ctx:
                dk.load_evalset(path)
            self.assertIn(":1:", str(ctx.exception))
        finally:
            os.unlink(path)


class TestCli(unittest.TestCase):
    def test_version(self):
        with self.assertRaises(SystemExit) as ctx:
            dk.main(["--version"])
        self.assertEqual(ctx.exception.code, 0)


if __name__ == "__main__":
    unittest.main()

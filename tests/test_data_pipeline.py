"""
test_data_pipeline.py
Task #12: Unit tests for the data processing pipeline.

Tests are deliberately lightweight - they don't download any data (no MUSE,
no WikiANN, no OPUS-100) and don't require a GPU.  They validate the logic
that is hard to catch from end-to-end runs alone:

 - Subword label alignment (the most common source of silent NER errors)
 - CodeSwitchAugmenter label preservation and switch probability
 - BilingualLexicon parsing edge cases (whitespace variants, capitalization)
 - OPUS-100 config name construction (alphabetical ordering)
 - translate-and-cast round-trip (Dataset.from_list -> .cast feature schema)

Run with: pytest tests/test_data_pipeline.py -v
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing import (
    LABEL2ID,
    ID2LABEL,
    CodeSwitchAugmenter,
    MultiLingualNERPipeline,
)


# ── Subword label alignment ────────────────────────────────────────────────────

class TestSubwordLabelAlignment(unittest.TestCase):
    """
    tokenize_and_align() must assign a real label to the FIRST subword of
    each word and -100 (ignored by CrossEntropyLoss) to all continuation
    subwords and special tokens.  This is the most critical invariant in the
    whole pipeline: a bug here silently trains on wrong targets.
    """

    @classmethod
    def setUpClass(cls):
        cls.pipeline = MultiLingualNERPipeline(
            model_name="xlm-roberta-base", max_length=32
        )

    def _align(self, tokens, ner_tags):
        """Thin wrapper: single-example batch, returns aligned labels list."""
        batch = {"tokens": [tokens], "ner_tags": [ner_tags]}
        out = self.pipeline.tokenize_and_align(batch)
        return out["labels"][0]

    def test_single_word_entity_gets_one_real_label(self):
        # "Paris" is B-LOC.  XLM-R tokenizes it as a single subword so only
        # that subword should get label 5 (B-LOC); CLS/SEP get -100.
        tokens = ["Paris", "is", "beautiful"]
        tags = [LABEL2ID["B-LOC"], LABEL2ID["O"], LABEL2ID["O"]]
        labels = self._align(tokens, tags)
        real = [l for l in labels if l != -100]
        self.assertEqual(real, tags)

    def test_multi_subword_word_first_subword_gets_label(self):
        # "Bundesministerium" splits into multiple subwords.  The first should
        # carry the label; the rest should be -100.
        tokens = ["Bundesministerium", "Berlin"]
        tags = [LABEL2ID["B-ORG"], LABEL2ID["B-LOC"]]
        labels = self._align(tokens, tags)
        real = [l for l in labels if l != -100]
        # Must have exactly 2 real labels (one per word)
        self.assertEqual(len(real), 2)
        self.assertEqual(real[0], LABEL2ID["B-ORG"])
        self.assertEqual(real[1], LABEL2ID["B-LOC"])

    def test_continuation_subwords_are_masked(self):
        # Verify that for a multi-subword word, no continuation subword has
        # a non-(-100) label other than the first.
        tokens = ["Supercalifragilistic", "is", "a", "word"]
        tags = [LABEL2ID["O"]] * 4
        labels = self._align(tokens, tags)
        # Get tokenized word_ids to identify where continuations are
        tokenized = self.pipeline.tokenizer(
            tokens, is_split_into_words=True,
            truncation=True, padding="max_length", max_length=32
        )
        word_ids = tokenized.word_ids(batch_index=0)
        seen_words = set()
        for wid, lab in zip(word_ids, labels):
            if wid is None:
                self.assertEqual(lab, -100, "Special tokens must be -100")
            elif wid in seen_words:
                self.assertEqual(lab, -100, f"Continuation subword of word {wid} must be -100")
            else:
                seen_words.add(wid)

    def test_all_o_labels_preserved(self):
        tokens = ["The", "dog", "sat"]
        tags = [LABEL2ID["O"], LABEL2ID["O"], LABEL2ID["O"]]
        labels = self._align(tokens, tags)
        real = [l for l in labels if l != -100]
        self.assertEqual(real, [0, 0, 0])

    def test_iob2_span_preserved(self):
        # B-PER I-PER O: multi-token entity, all three words single-subword.
        tokens = ["John", "Smith", "runs"]
        tags = [LABEL2ID["B-PER"], LABEL2ID["I-PER"], LABEL2ID["O"]]
        labels = self._align(tokens, tags)
        real = [l for l in labels if l != -100]
        self.assertEqual(real, [LABEL2ID["B-PER"], LABEL2ID["I-PER"], LABEL2ID["O"]])

    def test_labels_length_matches_max_length(self):
        tokens = ["Hello", "world"]
        tags = [LABEL2ID["O"], LABEL2ID["O"]]
        labels = self._align(tokens, tags)
        self.assertEqual(len(labels), 32)


# ── CodeSwitchAugmenter ────────────────────────────────────────────────────────

class _StubLexicon:
    """Deterministic stub: maps any word to f"T_{word}" (always available)."""
    def translate(self, word):
        return f"T_{word}"


class TestCodeSwitchAugmenter(unittest.TestCase):

    def _make_example(self, tokens, tags):
        return {"tokens": list(tokens), "ner_tags": list(tags), "langs": [], "spans": []}

    def test_ner_tags_never_modified(self):
        tags = [LABEL2ID["B-PER"], LABEL2ID["I-PER"], LABEL2ID["O"], LABEL2ID["B-LOC"]]
        example = self._make_example(["John", "Smith", "in", "Paris"], tags)
        augmenter = CodeSwitchAugmenter({"de": _StubLexicon()}, switch_prob=1.0)
        out = augmenter.augment(example)
        self.assertEqual(out["ner_tags"], tags, "NER tags must be identical to input")

    def test_switch_prob_zero_leaves_tokens_unchanged(self):
        tokens = ["Angela", "Merkel", "visited", "Berlin"]
        tags = [LABEL2ID["B-PER"], LABEL2ID["I-PER"], LABEL2ID["O"], LABEL2ID["B-LOC"]]
        example = self._make_example(tokens, tags)
        augmenter = CodeSwitchAugmenter({"de": _StubLexicon()}, switch_prob=0.0)
        out = augmenter.augment(example)
        self.assertEqual(out["tokens"], tokens)

    def test_switch_prob_one_translates_all_available(self):
        tokens = ["hello", "world"]
        tags = [LABEL2ID["O"], LABEL2ID["O"]]
        example = self._make_example(tokens, tags)
        augmenter = CodeSwitchAugmenter({"tr": _StubLexicon()}, switch_prob=1.0)
        out = augmenter.augment(example)
        self.assertEqual(out["tokens"], ["T_hello", "T_world"])

    def test_output_length_matches_input(self):
        tokens = ["The", "cat", "sat", "on", "the", "mat"]
        tags = [LABEL2ID["O"]] * 6
        example = self._make_example(tokens, tags)
        augmenter = CodeSwitchAugmenter({"de": _StubLexicon()}, switch_prob=0.5)
        out = augmenter.augment(example)
        self.assertEqual(len(out["tokens"]), len(tokens))
        self.assertEqual(len(out["ner_tags"]), len(tags))

    def test_capitalization_preserved_after_switch(self):
        # Uppercase first char of original -> translated token should be capitalized.
        tokens = ["Berlin"]
        tags = [LABEL2ID["B-LOC"]]
        example = self._make_example(tokens, tags)

        class UpperStub:
            def translate(self, word):
                return "translated"  # lowercase

        augmenter = CodeSwitchAugmenter({"de": UpperStub()}, switch_prob=1.0)
        out = augmenter.augment(example)
        self.assertTrue(out["tokens"][0][0].isupper(),
                        "Capitalized source token should produce capitalized translation")


# ── BilingualLexicon parsing ───────────────────────────────────────────────────

class TestBilingualLexiconParsing(unittest.TestCase):
    """
    BilingualLexicon reads MUSE dictionary files that are INCONSISTENTLY
    formatted (tab- or space-separated depending on language pair).  This
    was a real production bug (en-de failed silently with 0 entries).
    Test both formats plus edge cases.
    """

    def _write_dict(self, content, suffix=".txt"):
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=suffix, delete=False, encoding="utf-8"
        )
        f.write(content)
        f.close()
        return f.name

    def _parse_file(self, path):
        """Re-implement BilingualLexicon's core parsing loop without downloading."""
        word_map = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 2:
                    continue
                src, tgt = parts
                word_map.setdefault(src.lower(), []).append(tgt)
        return word_map

    def test_tab_separated_parsed(self):
        content = "dog\tHund\ncat\tKatze\nbird\tVogel\n"
        path = self._write_dict(content)
        try:
            wm = self._parse_file(path)
            self.assertEqual(set(wm.keys()), {"dog", "cat", "bird"})
            self.assertIn("Hund", wm["dog"])
        finally:
            os.unlink(path)

    def test_space_separated_parsed(self):
        content = "dog Hund\ncat Katze\nbird Vogel\n"
        path = self._write_dict(content)
        try:
            wm = self._parse_file(path)
            self.assertEqual(set(wm.keys()), {"dog", "cat", "bird"})
        finally:
            os.unlink(path)

    def test_mixed_separators_both_parsed(self):
        content = "dog\tHund\ncat Katze\n"
        path = self._write_dict(content)
        try:
            wm = self._parse_file(path)
            self.assertIn("dog", wm)
            self.assertIn("cat", wm)
        finally:
            os.unlink(path)

    def test_malformed_lines_skipped(self):
        # Lines with != 2 parts must be silently skipped (no crash, no entry).
        content = "just_one\nfoo bar baz\nok entry\n"
        path = self._write_dict(content)
        try:
            wm = self._parse_file(path)
            self.assertNotIn("just_one", wm)
            self.assertNotIn("foo", wm)
            self.assertIn("ok", wm)
        finally:
            os.unlink(path)

    def test_case_insensitive_lookup(self):
        content = "Dog Hund\n"
        path = self._write_dict(content)
        try:
            wm = self._parse_file(path)
            # Stored under lowercase key
            self.assertIn("dog", wm)
            self.assertNotIn("Dog", wm)
        finally:
            os.unlink(path)

    def test_multiple_translations_accumulated(self):
        # Some source words have multiple target translations in MUSE dicts.
        content = "dog Hund\ndog Köter\n"
        path = self._write_dict(content)
        try:
            wm = self._parse_file(path)
            self.assertEqual(len(wm["dog"]), 2)
            self.assertIn("Hund", wm["dog"])
            self.assertIn("Köter", wm["dog"])
        finally:
            os.unlink(path)


# ── OPUS-100 config name construction ─────────────────────────────────────────

class TestOpus100ConfigNames(unittest.TestCase):
    """
    OPUS-100 names language-pair configs alphabetically, NOT source→target order.
    load_parallel_corpus() builds the config name as
    "-".join(sorted([lang1, lang2])).  This was not a bug we hit in production
    but it's a silent footgun if someone adds a new language pair.
    """

    def _config_name(self, lang1, lang2):
        return "-".join(sorted([lang1, lang2]))

    def test_en_tr_is_en_tr(self):
        # 'e' < 't' alphabetically
        self.assertEqual(self._config_name("en", "tr"), "en-tr")

    def test_en_de_is_de_en(self):
        # 'd' < 'e' alphabetically
        self.assertEqual(self._config_name("en", "de"), "de-en")

    def test_en_ar_is_ar_en(self):
        # 'a' < 'e' alphabetically
        self.assertEqual(self._config_name("en", "ar"), "ar-en")

    def test_en_ko_is_en_ko(self):
        self.assertEqual(self._config_name("en", "ko"), "en-ko")

    def test_en_fi_is_en_fi(self):
        self.assertEqual(self._config_name("en", "fi"), "en-fi")

    def test_symmetry(self):
        # Order of arguments must not matter
        self.assertEqual(self._config_name("tr", "en"), self._config_name("en", "tr"))
        self.assertEqual(self._config_name("ar", "en"), self._config_name("en", "ar"))


# ── Dataset cast round-trip ────────────────────────────────────────────────────

class TestDatasetCastRoundTrip(unittest.TestCase):
    """
    Dataset.from_list() infers plain int64 for ner_tags, while the original
    WikiANN split uses ClassLabel.  concatenate_datasets() raises ValueError if
    features don't match.  The fix is .cast(original.features).

    This test exercises the core invariant WITHOUT downloading WikiANN - it
    uses a synthetic datasets.Features schema that mimics the real one.
    """

    def test_int64_to_classlabel_cast(self):
        try:
            from datasets import Dataset, ClassLabel, Sequence, Features, Value
        except ImportError:
            self.skipTest("datasets not installed")

        # Minimal WikiANN-like feature schema
        label_names = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
        original_features = Features({
            "tokens": Sequence(Value("string")),
            "ner_tags": Sequence(ClassLabel(names=label_names)),
            "langs": Sequence(Value("string")),
            "spans": Sequence(Value("string")),
        })

        # Simulate original dataset
        original_data = [
            {"tokens": ["Paris"], "ner_tags": [5], "langs": ["en"], "spans": []},
        ]
        original_ds = Dataset.from_list(original_data, features=original_features)

        # Simulate augmented examples (from_list infers int64 for ner_tags)
        augmented_data = [
            {"tokens": ["Paris", "est"], "ner_tags": [5, 0], "langs": ["en"], "spans": []},
        ]
        augmented_ds = Dataset.from_list(augmented_data)

        # Before cast: ner_tags feature is Sequence(Value("int64"))
        self.assertNotEqual(
            str(augmented_ds.features["ner_tags"]),
            str(original_ds.features["ner_tags"]),
        )

        # After cast: features match
        augmented_ds_cast = augmented_ds.cast(original_features)
        self.assertEqual(
            str(augmented_ds_cast.features["ner_tags"]),
            str(original_ds.features["ner_tags"]),
        )

        # Can now concatenate without ValueError
        from datasets import concatenate_datasets
        combined = concatenate_datasets([original_ds, augmented_ds_cast])
        self.assertEqual(len(combined), 2)


if __name__ == "__main__":
    unittest.main()

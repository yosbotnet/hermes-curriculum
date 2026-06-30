"""Tests for the OKF frontmatter codec and the file-backed content repository.

Stdlib unittest only. Everything here is deterministic: the frontmatter codec is
pure, and the file repository is exercised against a throwaway
``tempfile.TemporaryDirectory`` so no fixture state leaks between tests.

The invariants under test are:
* ``parse``/``dump`` are inverses for the supported YAML subset (scalars of every
  type, a flat inline list) and ``dump`` preserves the body verbatim;
* a document with no frontmatter parses to ``({}, text)``;
* ``FileContentRepository`` round-trips concepts (including their source-ref
  citations) and questions through real files;
* the returned hash is the SHA-256 of the file bytes, so it changes with the body
  and matches what ``iter_concepts`` reports;
* ``iter_concepts`` enumerates concepts but skips the reserved ``index.md`` /
  ``log.md`` files and the whole ``questions/`` subtree.
"""
from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from curriculum.domain.entities import ConceptContent, QuestionContent, SourceRef
from curriculum.okf import frontmatter
from curriculum.storage.okf_content import FileContentRepository


class FrontmatterParseTest(unittest.TestCase):
    def test_missing_frontmatter_returns_empty_meta_and_text(self) -> None:
        text = "no fence here\njust body\n"
        self.assertEqual(frontmatter.parse(text), ({}, text))

    def test_unterminated_block_is_treated_as_no_frontmatter(self) -> None:
        # A leading '---' with no closing fence is not a frontmatter block.
        text = "---\ntype: Concept\nstill no close\n"
        self.assertEqual(frontmatter.parse(text), ({}, text))

    def test_parses_scalars_of_each_type(self) -> None:
        text = (
            "---\n"
            "type: Concept\n"
            "count: 3\n"
            "weight: 0.5\n"
            "active: true\n"
            "off: False\n"
            "---\n"
            "body\n"
        )
        meta, body = frontmatter.parse(text)
        self.assertEqual(meta["type"], "Concept")
        self.assertEqual(meta["count"], 3)
        self.assertEqual(meta["weight"], 0.5)
        self.assertIs(meta["active"], True)
        self.assertIs(meta["off"], False)
        self.assertEqual(body, "body\n")

    def test_parses_inline_list(self) -> None:
        meta, _ = frontmatter.parse("---\ntags: [a, b, c]\n---\n")
        self.assertEqual(meta["tags"], ["a", "b", "c"])

    def test_empty_inline_list(self) -> None:
        meta, _ = frontmatter.parse("---\ntags: []\n---\n")
        self.assertEqual(meta["tags"], [])

    def test_value_after_first_colon_is_kept_intact(self) -> None:
        # Colons inside a value (e.g. a URL) must not be split.
        meta, _ = frontmatter.parse("---\nresource: http://x/y:z\n---\n")
        self.assertEqual(meta["resource"], "http://x/y:z")

    def test_comments_and_blank_lines_ignored(self) -> None:
        meta, _ = frontmatter.parse("---\n# a comment\n\ntype: Concept\n---\n")
        self.assertEqual(meta, {"type": "Concept"})

    def test_quoted_string_keeps_numeric_text_as_string(self) -> None:
        meta, _ = frontmatter.parse('---\nver: "0.1"\n---\n')
        self.assertEqual(meta["ver"], "0.1")
        self.assertIsInstance(meta["ver"], str)


class FrontmatterRoundTripTest(unittest.TestCase):
    def assert_round_trip(self, meta: dict, body: str) -> None:
        rendered = frontmatter.dump(meta, body)
        self.assertEqual(frontmatter.parse(rendered), (meta, body))

    def test_round_trip_mixed_scalars_and_list(self) -> None:
        self.assert_round_trip(
            {
                "type": "Concept",
                "title": "Sorting",
                "tags": ["sort", "order"],
                "count": 7,
                "weight": 0.25,
                "active": True,
            },
            "# Body\n\nSome prose.\n",
        )

    def test_round_trip_empty_meta_returns_body_verbatim(self) -> None:
        body = "plain document, no frontmatter\n"
        self.assertEqual(frontmatter.dump({}, body), body)
        self.assert_round_trip({}, body)

    def test_round_trip_preserves_empty_body(self) -> None:
        self.assert_round_trip({"type": "Concept"}, "")

    def test_round_trip_preserves_body_without_trailing_newline(self) -> None:
        self.assert_round_trip({"type": "Concept"}, "no trailing newline")

    def test_round_trip_quotes_ambiguous_strings(self) -> None:
        # Strings that look like other types, are empty, or lead with an
        # indicator char must be quoted so they read back as the same string.
        self.assert_round_trip(
            {
                "type": "Concept",
                "looks_bool": "true",
                "looks_int": "42",
                "leading_bracket": "[draft]",
                "empty": "",
                "padded": " spaced ",
            },
            "b\n",
        )

    def test_round_trip_list_with_comma_element(self) -> None:
        self.assert_round_trip({"items": ["a,b", "c"]}, "x")

    def test_round_trip_body_containing_fence_line(self) -> None:
        # A body that itself contains a '---' line must survive: the first
        # closing fence ends the metadata, the rest is body.
        self.assert_round_trip({"type": "Concept"}, "intro\n---\nmore\n")


class FileConceptRepositoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.repo = FileContentRepository(self.root)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_get_missing_concept_returns_none(self) -> None:
        self.assertIsNone(self.repo.get_concept_content("absent"))

    def test_concept_round_trip_with_source_refs(self) -> None:
        content = ConceptContent(
            concept_id="algorithms/sorting",
            title="Sorting",
            body="# Overview\n\nSorting orders elements.",
            description="How sorting works",
            source_refs=(SourceRef(file="notes.md", line=12), SourceRef(file="ch1.md")),
        )
        self.repo.put_concept_content(content)
        self.assertEqual(self.repo.get_concept_content("algorithms/sorting"), content)

    def test_concept_round_trip_without_source_refs(self) -> None:
        content = ConceptContent(
            concept_id="intro",
            title="Intro",
            body="# Hello\n\nNo citations here.",
            description="",
        )
        self.repo.put_concept_content(content)
        self.assertEqual(self.repo.get_concept_content("intro"), content)

    def test_concept_file_has_frontmatter_type(self) -> None:
        self.repo.put_concept_content(
            ConceptContent(concept_id="c", title="C", body="b")
        )
        text = (self.root / "c.md").read_text(encoding="utf-8")
        meta, _ = frontmatter.parse(text)
        self.assertEqual(meta["type"], "Concept")

    def test_put_creates_nested_parent_dirs(self) -> None:
        self.repo.put_concept_content(
            ConceptContent(concept_id="a/b/c", title="Deep", body="x")
        )
        self.assertTrue((self.root / "a" / "b" / "c.md").is_file())

    def test_put_returns_sha256_of_file_bytes(self) -> None:
        digest = self.repo.put_concept_content(
            ConceptContent(concept_id="c", title="C", body="body")
        )
        on_disk = (self.root / "c.md").read_bytes()
        self.assertEqual(digest, hashlib.sha256(on_disk).hexdigest())
        self.assertEqual(len(digest), 64)

    def test_hash_changes_when_body_changes(self) -> None:
        h1 = self.repo.put_concept_content(
            ConceptContent(concept_id="c", title="C", body="one")
        )
        h2 = self.repo.put_concept_content(
            ConceptContent(concept_id="c", title="C", body="two")
        )
        self.assertNotEqual(h1, h2)

    def test_source_ref_without_line_round_trips(self) -> None:
        content = ConceptContent(
            concept_id="c",
            title="C",
            body="b",
            source_refs=(SourceRef(file="only-file.md"),),
        )
        self.repo.put_concept_content(content)
        loaded = self.repo.get_concept_content("c")
        self.assertEqual(loaded.source_refs, (SourceRef(file="only-file.md", line=None),))


class FileQuestionRepositoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.repo = FileContentRepository(self.root)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_get_missing_question_returns_none(self) -> None:
        self.assertIsNone(self.repo.get_question_content("absent"))

    def test_question_round_trip_with_rubric(self) -> None:
        content = QuestionContent(
            question_id="q1",
            prompt="What is a stable sort?",
            rubric="Mentions that equal keys keep their order.",
        )
        self.repo.put_question_content(content)
        self.assertEqual(self.repo.get_question_content("q1"), content)

    def test_question_round_trip_without_rubric(self) -> None:
        content = QuestionContent(question_id="q2", prompt="Define a heap.")
        self.repo.put_question_content(content)
        self.assertEqual(self.repo.get_question_content("q2"), content)

    def test_question_stored_under_questions_subtree(self) -> None:
        self.repo.put_question_content(QuestionContent(question_id="q1", prompt="p"))
        self.assertTrue((self.root / "questions" / "q1.md").is_file())

    def test_question_file_has_frontmatter_type(self) -> None:
        self.repo.put_question_content(QuestionContent(question_id="q1", prompt="p"))
        meta, _ = frontmatter.parse(
            (self.root / "questions" / "q1.md").read_text(encoding="utf-8")
        )
        self.assertEqual(meta["type"], "Question")


class IterConceptsTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.repo = FileContentRepository(self.root)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_empty_bundle_yields_nothing(self) -> None:
        self.assertEqual(list(self.repo.iter_concepts()), [])

    def test_finds_concepts_and_skips_reserved_and_questions(self) -> None:
        h_intro = self.repo.put_concept_content(
            ConceptContent(concept_id="intro", title="Intro", body="i")
        )
        h_sort = self.repo.put_concept_content(
            ConceptContent(concept_id="algorithms/sorting", title="Sort", body="s")
        )
        # A question lives under questions/ and must NOT appear as a concept.
        self.repo.put_question_content(QuestionContent(question_id="q1", prompt="p"))
        # Reserved files at two levels must be skipped.
        (self.root / "index.md").write_text("# root index\n", encoding="utf-8")
        (self.root / "log.md").write_text("# log\n", encoding="utf-8")
        (self.root / "algorithms" / "index.md").write_text("# idx\n", encoding="utf-8")

        pairs = list(self.repo.iter_concepts())
        # Sorted by concept id, only the two real concepts, with file-byte hashes.
        self.assertEqual(
            pairs, [("algorithms/sorting", h_sort), ("intro", h_intro)]
        )

    def test_iter_hash_matches_put_return(self) -> None:
        h = self.repo.put_concept_content(
            ConceptContent(concept_id="c", title="C", body="b")
        )
        self.assertEqual(dict(self.repo.iter_concepts())["c"], h)

    def test_iter_reflects_updated_hash(self) -> None:
        self.repo.put_concept_content(ConceptContent(concept_id="c", title="C", body="old"))
        new = self.repo.put_concept_content(
            ConceptContent(concept_id="c", title="C", body="new")
        )
        self.assertEqual(dict(self.repo.iter_concepts())["c"], new)


if __name__ == "__main__":
    unittest.main()

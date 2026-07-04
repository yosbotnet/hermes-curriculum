"""Tests for the corpus-preparation onboarding tools (curriculum.app.corpus_tools).

These pin the two halves of the onboarding bridge that turns raw course
materials into a buildable corpus:

* ``scaffold`` -- writes the three starter artifacts (a ``materials/`` dir, a
  commented ``corpus.json`` template, and a ``materials/README.txt``) and REFUSES
  to clobber an existing manifest, so re-running it never destroys a corpus a user
  has already begun filling in.
* ``validate`` -- a machine-checkable report an agent can loop against: it flags
  the concrete ways a source file is not yet ingestable (missing, still-binary
  PDF/zip, non-UTF-8, empty, suspiciously short), warns when no trusted spine is
  declared, and estimates the extract-call cost BEFORE any paid inference runs.

Everything here is stdlib-only and strictly offline: ``validate`` reads files
from a temp dir and never opens a socket or a database.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from curriculum.app import build, corpus_tools


class ScaffoldTest(unittest.TestCase):
    def test_scaffold_creates_the_three_artifacts(self) -> None:
        with TemporaryDirectory() as directory:
            created = corpus_tools.scaffold(directory)
            root = Path(directory)
            manifest = root / "corpus.json"
            materials = root / "materials"
            readme = materials / "README.txt"
            self.assertTrue(materials.is_dir(), "materials/ dir must be created")
            self.assertTrue(manifest.is_file(), "corpus.json must be created")
            self.assertTrue(readme.is_file(), "materials/README.txt must be created")
            # The returned paths point at exactly what was written.
            created_set = {str(Path(p)) for p in created}
            self.assertIn(str(manifest), created_set)
            self.assertIn(str(readme), created_set)

    def test_scaffold_manifest_is_a_commented_template_with_spine_and_satellite(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            corpus_tools.scaffold(directory)
            data = json.loads((Path(directory) / "corpus.json").read_text("utf-8"))
            # Uses "_comment" keys like corpus.example.json.
            self.assertIn("_comment", data)
            self.assertIsInstance(data.get("sources"), list)
            self.assertGreaterEqual(len(data["sources"]), 2)
            spine = [s for s in data["sources"] if s.get("spine") is True]
            satellites = [s for s in data["sources"] if not s.get("spine")]
            self.assertTrue(spine, "one example source must be spine:true")
            self.assertTrue(satellites, "one example satellite source")
            # Example paths point under materials/ and do not exist yet.
            for source in data["sources"]:
                self.assertTrue(source["path"].startswith("materials/"))
                self.assertFalse((Path(directory) / source["path"]).exists())

    def test_scaffold_refuses_to_overwrite_existing_manifest(self) -> None:
        with TemporaryDirectory() as directory:
            manifest = Path(directory) / "corpus.json"
            manifest.write_text('{"mine": true}', encoding="utf-8")
            with self.assertRaises(Exception) as ctx:
                corpus_tools.scaffold(directory)
            self.assertIn("corpus.json", str(ctx.exception))
            # The user's file is untouched.
            self.assertEqual(
                json.loads(manifest.read_text("utf-8")), {"mine": True}
            )

    def test_scaffold_readme_is_plain_ascii_and_nonempty(self) -> None:
        with TemporaryDirectory() as directory:
            corpus_tools.scaffold(directory)
            text = (Path(directory) / "materials" / "README.txt").read_text("utf-8")
            text.encode("ascii")  # ASCII-only, no emojis/special chars
            self.assertGreaterEqual(len(text.splitlines()), 5)


def _write_manifest(root: Path, sources: list[dict], chunk_lines: int = 150) -> Path:
    manifest = root / "corpus.json"
    manifest.write_text(
        json.dumps({"course": "C", "chunk_lines": chunk_lines, "sources": sources}),
        encoding="utf-8",
    )
    return manifest


class ValidateReportShapeTest(unittest.TestCase):
    def test_report_has_the_four_sections(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.txt").write_text("hello " * 100, encoding="utf-8")
            manifest = _write_manifest(root, [{"path": "a.txt", "token": "a"}])
            report = corpus_tools.validate(str(manifest))
            for key in ("errors", "warnings", "sources", "estimate"):
                self.assertIn(key, report)
            self.assertIsInstance(report["errors"], list)
            self.assertIsInstance(report["warnings"], list)
            self.assertIsInstance(report["sources"], list)
            self.assertIsInstance(report["estimate"], dict)

    def test_bad_manifest_shape_becomes_an_error_not_an_exception(self) -> None:
        with TemporaryDirectory() as directory:
            manifest = Path(directory) / "corpus.json"
            manifest.write_text('{"course": "", "sources": []}', encoding="utf-8")
            report = corpus_tools.validate(str(manifest))
            self.assertTrue(report["errors"], "ConfigError must land in errors")


class ValidatePerSourceErrorTest(unittest.TestCase):
    def test_missing_file_is_an_error(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _write_manifest(root, [{"path": "nope.txt", "token": "n"}])
            report = corpus_tools.validate(str(manifest))
            blob = " ".join(report["errors"]).lower()
            self.assertTrue(report["errors"])
            self.assertIn("nope.txt", " ".join(report["errors"]))
            self.assertIn("missing", blob)

    def test_pdf_magic_is_an_error_with_a_pointed_hint(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "slides.pdf.txt").write_bytes(b"%PDF-1.4\n binary junk here")
            manifest = _write_manifest(
                root, [{"path": "slides.pdf.txt", "token": "s"}]
            )
            report = corpus_tools.validate(str(manifest))
            blob = " ".join(report["errors"]).lower()
            self.assertTrue(report["errors"])
            self.assertIn("pdf", blob)
            self.assertIn("extract", blob)

    def test_zip_magic_is_an_error_with_a_pointed_hint(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "notes.txt").write_bytes(b"PK\x03\x04 rest of a docx zip")
            manifest = _write_manifest(root, [{"path": "notes.txt", "token": "n"}])
            report = corpus_tools.validate(str(manifest))
            blob = " ".join(report["errors"]).lower()
            self.assertTrue(report["errors"])
            self.assertIn("zip", blob)
            self.assertIn("extract", blob)

    def test_null_byte_binary_is_an_error(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "bin.txt").write_bytes(b"some text\x00then a null byte")
            manifest = _write_manifest(root, [{"path": "bin.txt", "token": "b"}])
            report = corpus_tools.validate(str(manifest))
            blob = " ".join(report["errors"]).lower()
            self.assertTrue(report["errors"])
            self.assertIn("binary", blob)

    def test_non_utf8_is_an_error(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "latin.txt").write_bytes(b"\xff\xfe invalid utf-8 bytes here")
            manifest = _write_manifest(root, [{"path": "latin.txt", "token": "l"}])
            report = corpus_tools.validate(str(manifest))
            self.assertTrue(report["errors"])
            self.assertIn("utf-8", " ".join(report["errors"]).lower())

    def test_empty_file_is_an_error(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "empty.txt").write_bytes(b"")
            manifest = _write_manifest(root, [{"path": "empty.txt", "token": "e"}])
            report = corpus_tools.validate(str(manifest))
            self.assertTrue(report["errors"])
            self.assertIn("empty", " ".join(report["errors"]).lower())


class ValidatePendingProcurementTest(unittest.TestCase):
    def test_missing_file_with_procure_is_pending_not_error(self) -> None:
        # A declared-but-unfilled slot (missing file + procure instruction) must
        # not block the build: it is a distinct "pending" status and a warning
        # carrying the procurement instruction, so the agent's zero-errors loop
        # passes and the user knows exactly what to go get.
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.txt").write_text("teaching text " * 50, encoding="utf-8")
            manifest = _write_manifest(
                root,
                [
                    {"path": "a.txt", "token": "a"},
                    {"path": "nope.txt", "token": "n",
                     "procure": "extract chapter 3 of the owned textbook"},
                ],
            )
            report = corpus_tools.validate(str(manifest))
        self.assertEqual(report["errors"], [])
        rows = {row["token"]: row for row in report["sources"]}
        self.assertEqual(rows["n"]["status"], "pending")
        self.assertEqual(rows["a"]["status"], "ok")
        blob = " ".join(report["warnings"])
        self.assertIn("pending procurement", blob)
        self.assertIn("extract chapter 3 of the owned textbook", blob)

    def test_present_file_with_procure_is_ordinary(self) -> None:
        # Once the user drops the file in, the procure note is inert: the slot
        # is filled and validates like any other source.
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.txt").write_text("teaching text " * 50, encoding="utf-8")
            manifest = _write_manifest(
                root,
                [{"path": "a.txt", "token": "a", "procure": "already done"}],
            )
            report = corpus_tools.validate(str(manifest))
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["sources"][0]["status"], "ok")
        self.assertNotIn("pending procurement", " ".join(report["warnings"]))

    def test_pending_spine_warns_pending_not_the_generic_no_spine_advice(self) -> None:
        # A spine slot that is still pending supplies no ordering for THIS
        # build, so a warning must fire -- but the pointed one ("your spine is
        # still pending"), not the generic "consider marking spine:true"
        # advice, which would be misleading when a spine IS declared.
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.txt").write_text("teaching text " * 50, encoding="utf-8")
            manifest = _write_manifest(
                root,
                [
                    {"path": "a.txt", "token": "a"},
                    {"path": "spine.txt", "token": "s", "spine": True,
                     "procure": "extract the textbook chapters"},
                ],
            )
            report = corpus_tools.validate(str(manifest))
        self.assertEqual(report["errors"], [])
        blob = " ".join(report["warnings"])
        self.assertIn("spine", blob)
        self.assertIn("pending", blob)
        self.assertNotIn("consider marking a trusted ordering", blob)


class ValidateWarningTest(unittest.TestCase):
    def test_short_file_is_a_warning_not_an_error(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tiny.txt").write_text("too short", encoding="utf-8")
            manifest = _write_manifest(
                root, [{"path": "tiny.txt", "token": "t", "spine": True}]
            )
            report = corpus_tools.validate(str(manifest))
            self.assertFalse(report["errors"], "a short file is only a warning")
            blob = " ".join(report["warnings"]).lower()
            self.assertIn("short", blob)

    def test_no_spine_source_is_a_warning(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.txt").write_text("word " * 100, encoding="utf-8")
            (root / "b.txt").write_text("word " * 100, encoding="utf-8")
            manifest = _write_manifest(
                root,
                [
                    {"path": "a.txt", "token": "a"},
                    {"path": "b.txt", "token": "b"},
                ],
            )
            report = corpus_tools.validate(str(manifest))
            blob = " ".join(report["warnings"]).lower()
            self.assertIn("spine", blob)

    def test_spine_present_means_no_spine_warning(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.txt").write_text("word " * 100, encoding="utf-8")
            manifest = _write_manifest(
                root, [{"path": "a.txt", "token": "a", "spine": True}]
            )
            report = corpus_tools.validate(str(manifest))
            spine_warnings = [w for w in report["warnings"] if "spine" in w.lower()]
            self.assertFalse(spine_warnings)

    def test_short_spine_source_does_not_trigger_no_spine_warning(self) -> None:
        # A spine source whose ONLY defect is the short-file warning still gives
        # the build a usable ordering (it will be chained), so it must NOT trip
        # the spurious "no spine source" warning.
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tiny.txt").write_text("too short", encoding="utf-8")
            manifest = _write_manifest(
                root, [{"path": "tiny.txt", "token": "t", "spine": True}]
            )
            report = corpus_tools.validate(str(manifest))
            spine_warnings = [
                w for w in report["warnings"] if "no spine source" in w.lower()
            ]
            self.assertFalse(spine_warnings, "short spine must not warn no-spine")

    def test_spine_source_with_error_does_not_count_as_spine(self) -> None:
        # A spine source that is itself in error must NOT suppress the no-spine
        # warning: there is no usable trusted ordering.
        with TemporaryDirectory() as directory:
            root = Path(directory)
            # spine file is missing -> error; the satellite is fine.
            (root / "b.txt").write_text("word " * 100, encoding="utf-8")
            manifest = _write_manifest(
                root,
                [
                    {"path": "gone.txt", "token": "g", "spine": True},
                    {"path": "b.txt", "token": "b"},
                ],
            )
            report = corpus_tools.validate(str(manifest))
            blob = " ".join(report["warnings"]).lower()
            self.assertIn("spine", blob)


class ValidateCleanAndEstimateTest(unittest.TestCase):
    def test_clean_two_source_corpus_validates_with_zero_errors(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            # 300 lines each, chunk_lines=150 -> 2 chunks per source, 4 total.
            body = "\n".join(f"line {i} with real teaching text" for i in range(300))
            (root / "spine.txt").write_text(body, encoding="utf-8")
            (root / "sat.txt").write_text(body, encoding="utf-8")
            manifest = _write_manifest(
                root,
                [
                    {"path": "spine.txt", "token": "spine", "spine": True},
                    {"path": "sat.txt", "token": "sat"},
                ],
                chunk_lines=150,
            )
            report = corpus_tools.validate(str(manifest))
            self.assertEqual(report["errors"], [])
            estimate = report["estimate"]
            self.assertEqual(estimate["chunks"], 4)
            self.assertEqual(estimate["extract_calls"], 4)
            # Per-source rows carry the line/chunk breakdown.
            self.assertEqual(len(report["sources"]), 2)
            for source in report["sources"]:
                self.assertEqual(source["lines"], 300)
                self.assertEqual(source["chunks"], 2)
                self.assertEqual(source["status"], "ok")


    def test_warning_status_source_chunks_are_counted_in_estimate(self) -> None:
        # A short-but-valid (warning) source shows chunks in its row; those chunks
        # WILL be ingested, so they must be added to the estimate totals -- not
        # gated out because the status is "warning" rather than "ok".
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tiny.txt").write_text("too short", encoding="utf-8")
            manifest = _write_manifest(
                root, [{"path": "tiny.txt", "token": "t", "spine": True}]
            )
            report = corpus_tools.validate(str(manifest))
            row = report["sources"][0]
            self.assertEqual(row["status"], "warning")
            self.assertGreater(row["chunks"], 0)
            # The warning source's chunks must be reflected in the estimate.
            self.assertEqual(report["estimate"]["chunks"], row["chunks"])
            self.assertEqual(report["estimate"]["extract_calls"], row["chunks"])


class ValidateResolutionParityTest(unittest.TestCase):
    """validate resolves source paths the same way the build reads them.

    The single source of truth is :func:`build.load_manifest`, which normalises
    each source path to an absolute path against the MANIFEST's directory. validate
    must trust that resolution (not re-resolve against CWD), so its report is
    parity-by-construction with what the build will actually ingest.
    """

    def test_validate_trusts_load_manifest_resolved_paths(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "materials").mkdir()
            body = "\n".join(f"line {i} teaching text" for i in range(300))
            (root / "materials" / "a.txt").write_text(body, encoding="utf-8")
            manifest = _write_manifest(
                root, [{"path": "materials/a.txt", "token": "a", "spine": True}]
            )
            # The file load_manifest resolves is exactly the one that exists.
            loaded = build.load_manifest(str(manifest))
            resolved = Path(loaded["sources"][0]["path"])
            self.assertTrue(resolved.is_absolute())
            self.assertTrue(resolved.is_file())
            # And validate agrees: no errors (parity by construction).
            report = corpus_tools.validate(str(manifest))
            self.assertEqual(report["errors"], [])

    def test_absolute_source_path_still_validates(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            body = "\n".join(f"line {i} teaching text" for i in range(300))
            target = root / "elsewhere.txt"
            target.write_text(body, encoding="utf-8")
            manifest = _write_manifest(
                root, [{"path": str(target), "token": "a", "spine": True}]
            )
            report = corpus_tools.validate(str(manifest))
            self.assertEqual(report["errors"], [])


if __name__ == "__main__":
    unittest.main()

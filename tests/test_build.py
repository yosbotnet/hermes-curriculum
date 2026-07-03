"""Tests for the config-driven build orchestration (curriculum.app.build).

Stdlib unittest only and deliberately offline. The manifest loader is pure
(JSON parse + validation), so it is unit-tested directly against throwaway temp
files -- happy path, the ``chunk_lines`` default, and every malformed shape that
must raise :class:`ConfigError`. The network/DB orchestration functions
(``ingest``/``link``/``generate_questions``/``status``) are NOT exercised here:
their collaborators -- the ingestion pipeline and the embedding linker -- own
those tests. What we DO assert is the load-bearing property that makes this
module safe to import everywhere: it imports with neither ``psycopg`` nor a
network present (the heavy collaborators are deferred to call time).
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from curriculum.app import build
from curriculum.domain.entities import Concept, SourceRef
from curriculum.domain.enums import EdgeType
from curriculum.domain.errors import ConfigError
from curriculum.storage.memory import (
    InMemoryConceptIndexRepository,
    InMemoryEdgeRepository,
)


def _write_manifest(directory: str, payload: object) -> str:
    """Write ``payload`` as JSON into ``directory`` and return its path."""
    path = Path(directory) / "corpus.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


class LoadManifestHappyPathTest(unittest.TestCase):
    def test_full_manifest_is_normalised(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = _write_manifest(
                directory,
                {
                    "course": "Cybersecurity",
                    "chunk_lines": 80,
                    "sources": [
                        {"path": "materials/a.txt", "token": "a"},
                        # Extra keys on a source are dropped by normalisation.
                        {"path": "materials/b.txt", "token": "b", "note": "ignored"},
                    ],
                },
            )
            manifest = build.load_manifest(path)
            # CONTRACT (deliberately updated): source paths are normalised to
            # ABSOLUTE paths resolved against the MANIFEST file's directory, so the
            # build (which may run from a different CWD) reads exactly the file the
            # validator checked. Absolute == manifest-dir-relative here.
            base = Path(path).resolve().parent

        self.assertEqual(manifest["course"], "Cybersecurity")
        self.assertEqual(manifest["chunk_lines"], 80)
        self.assertEqual(
            manifest["sources"],
            [
                {"path": str(base / "materials/a.txt"), "token": "a"},
                {"path": str(base / "materials/b.txt"), "token": "b"},
            ],
        )
        for source in manifest["sources"]:
            self.assertTrue(Path(source["path"]).is_absolute())

    def test_relative_source_paths_resolve_against_manifest_dir_not_cwd(self) -> None:
        # Cross-dir parity: load_manifest must resolve a relative source path
        # against the manifest's own directory, independent of the process CWD, so
        # the resolved path points at the file that actually exists on disk.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "materials").mkdir()
            (root / "materials" / "a.txt").write_text("teaching text", encoding="utf-8")
            path = _write_manifest(
                directory,
                {"course": "C", "sources": [{"path": "materials/a.txt", "token": "a"}]},
            )
            manifest = build.load_manifest(path)
            resolved = Path(manifest["sources"][0]["path"])
            self.assertTrue(resolved.is_absolute())
            self.assertTrue(
                resolved.is_file(), "resolved path must point at the real file"
            )
            self.assertEqual(resolved, root / "materials" / "a.txt")

    def test_absolute_source_paths_are_preserved_unchanged(self) -> None:
        # An absolute source path must survive normalisation intact (base / abs == abs).
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            absolute = str(root / "elsewhere.txt")
            path = _write_manifest(
                directory,
                {"course": "C", "sources": [{"path": absolute, "token": "a"}]},
            )
            manifest = build.load_manifest(path)
        self.assertEqual(manifest["sources"][0]["path"], str(Path(absolute)))

    def test_chunk_lines_defaults_to_150_when_absent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = _write_manifest(
                directory,
                {"course": "C", "sources": [{"path": "a.txt", "token": "a"}]},
            )
            manifest = build.load_manifest(path)
        self.assertEqual(manifest["chunk_lines"], 150)

    def test_example_manifest_shape_loads(self) -> None:
        # The shipped corpus.example.json shape (minus its leading _comment) must
        # be accepted, since it is the documented template users copy.
        with tempfile.TemporaryDirectory() as directory:
            path = _write_manifest(
                directory,
                {
                    "_comment": "irrelevant top-level extra",
                    "course": "YourCourse",
                    "chunk_lines": 150,
                    "sources": [
                        {"path": "materials/lecture-01-intro.txt", "token": "lecture-01-intro"}
                    ],
                },
            )
            manifest = build.load_manifest(path)
        self.assertEqual(manifest["course"], "YourCourse")
        self.assertEqual(len(manifest["sources"]), 1)


class LoadManifestMalformedTest(unittest.TestCase):
    def _assert_rejects(self, payload: object) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = _write_manifest(directory, payload)
            with self.assertRaises(ConfigError):
                build.load_manifest(path)

    def test_top_level_not_an_object(self) -> None:
        self._assert_rejects([{"path": "a", "token": "a"}])

    def test_missing_course(self) -> None:
        self._assert_rejects({"sources": [{"path": "a", "token": "a"}]})

    def test_blank_course(self) -> None:
        self._assert_rejects({"course": "   ", "sources": [{"path": "a", "token": "a"}]})

    def test_non_string_course(self) -> None:
        self._assert_rejects({"course": 1, "sources": [{"path": "a", "token": "a"}]})

    def test_missing_sources(self) -> None:
        self._assert_rejects({"course": "C"})

    def test_empty_sources(self) -> None:
        self._assert_rejects({"course": "C", "sources": []})

    def test_sources_not_a_list(self) -> None:
        self._assert_rejects({"course": "C", "sources": {"path": "a", "token": "a"}})

    def test_source_not_an_object(self) -> None:
        self._assert_rejects({"course": "C", "sources": ["a.txt"]})

    def test_source_missing_path(self) -> None:
        self._assert_rejects({"course": "C", "sources": [{"token": "a"}]})

    def test_source_missing_token(self) -> None:
        self._assert_rejects({"course": "C", "sources": [{"path": "a.txt"}]})

    def test_source_blank_token(self) -> None:
        self._assert_rejects({"course": "C", "sources": [{"path": "a.txt", "token": ""}]})

    def test_non_positive_chunk_lines(self) -> None:
        self._assert_rejects(
            {"course": "C", "sources": [{"path": "a", "token": "a"}], "chunk_lines": 0}
        )

    def test_boolean_chunk_lines_rejected(self) -> None:
        # bool is an int subclass; it must not slip through as a "valid integer".
        self._assert_rejects(
            {"course": "C", "sources": [{"path": "a", "token": "a"}], "chunk_lines": True}
        )

    def test_invalid_json_raises_config_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.json"
            path.write_text("{ not valid json", encoding="utf-8")
            with self.assertRaises(ConfigError):
                build.load_manifest(str(path))

    def test_missing_file_raises_config_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = str(Path(directory) / "does-not-exist.json")
            with self.assertRaises(ConfigError):
                build.load_manifest(missing)


def _concept(cid: str, line: int, source: str) -> Concept:
    """A concept grounded at ``source``:``line`` (a spine source's citation)."""
    return Concept(id=cid, course="C", title=cid, source_refs=(SourceRef(source, line),))


class SpineStitchTest(unittest.TestCase):
    """Cross-source spine stitching (``build._spine_stitch_edges``).

    A pure, offline unit: given the manifest sources (with their ``spine`` flag
    and ``token``) and the per-source-index lists of persisted concepts, it
    stitches consecutive spine sources -- in manifest order -- with exactly one
    ``tail(A) -> head(B)`` PREREQUISITE edge per adjacent spine pair.
    """

    def test_two_spine_sources_get_one_cross_edge_tail_to_head(self) -> None:
        sources = [
            {"path": "a.txt", "token": "chA", "spine": True},
            {"path": "b.txt", "token": "chB", "spine": True},
        ]
        # Concepts supplied OUT of document order so the assertion proves the
        # stitch sorts by first source_ref line (tail = a2@20, head = b1@10).
        concepts = {
            0: [_concept("C/a2", 20, "chA"), _concept("C/a1", 10, "chA")],
            1: [_concept("C/b2", 20, "chB"), _concept("C/b1", 10, "chB")],
        }
        edges = build._spine_stitch_edges(sources, concepts)

        self.assertEqual(len(edges), 1)
        edge = edges[0]
        self.assertEqual((edge.src, edge.dst), ("C/a2", "C/b1"))
        self.assertIs(edge.type, EdgeType.PREREQUISITE)
        self.assertEqual(edge.provenance, "spine")
        self.assertEqual(edge.confidence, 1.0)
        self.assertEqual(edge.rationale, "spine order: chA -> chB")
        self.assertIsNotNone(edge.source_ref)
        self.assertEqual(edge.source_ref.file, "chA")
        self.assertEqual(edge.source_ref.line, 20)  # the tail concept's first ref

    def test_satellite_between_two_spines_does_not_break_adjacency(self) -> None:
        sources = [
            {"path": "a.txt", "token": "chA", "spine": True},
            {"path": "s.txt", "token": "sat"},  # satellite: no spine flag
            {"path": "b.txt", "token": "chB", "spine": True},
        ]
        concepts = {
            0: [_concept("C/a1", 10, "chA"), _concept("C/a2", 20, "chA")],
            1: [_concept("C/s1", 10, "sat")],  # satellite concepts
            2: [_concept("C/b1", 10, "chB"), _concept("C/b2", 20, "chB")],
        }
        edges = build._spine_stitch_edges(sources, concepts)

        self.assertEqual(len(edges), 1)
        self.assertEqual((edges[0].src, edges[0].dst), ("C/a2", "C/b1"))
        # No stitch edge touches a satellite concept.
        touched = {e.src for e in edges} | {e.dst for e in edges}
        self.assertNotIn("C/s1", touched)

    def test_single_spine_source_makes_no_cross_edge(self) -> None:
        sources = [{"path": "a.txt", "token": "chA", "spine": True}]
        concepts = {0: [_concept("C/a1", 10, "chA"), _concept("C/a2", 20, "chA")]}
        self.assertEqual(build._spine_stitch_edges(sources, concepts), [])

    def test_zero_concept_spine_source_is_skipped_neighbors_stitched(self) -> None:
        sources = [
            {"path": "a.txt", "token": "chA", "spine": True},
            {"path": "e.txt", "token": "chEmpty", "spine": True},
            {"path": "b.txt", "token": "chB", "spine": True},
        ]
        concepts = {
            0: [_concept("C/a1", 10, "chA"), _concept("C/a2", 20, "chA")],
            1: [],  # produced zero concepts -> its neighbors become adjacent
            2: [_concept("C/b1", 10, "chB"), _concept("C/b2", 20, "chB")],
        }
        edges = build._spine_stitch_edges(sources, concepts)

        self.assertEqual(len(edges), 1)
        self.assertEqual((edges[0].src, edges[0].dst), ("C/a2", "C/b1"))
        self.assertEqual(edges[0].rationale, "spine order: chA -> chB")

    def test_missing_source_index_is_treated_as_zero_concepts(self) -> None:
        # A source that failed to ingest never lands in the index->concepts map;
        # it must be skipped exactly like an empty one.
        sources = [
            {"path": "a.txt", "token": "chA", "spine": True},
            {"path": "e.txt", "token": "chEmpty", "spine": True},
            {"path": "b.txt", "token": "chB", "spine": True},
        ]
        concepts = {
            0: [_concept("C/a1", 10, "chA")],
            2: [_concept("C/b1", 10, "chB")],
        }
        edges = build._spine_stitch_edges(sources, concepts)
        self.assertEqual(len(edges), 1)
        self.assertEqual((edges[0].src, edges[0].dst), ("C/a1", "C/b1"))

    def test_stitch_is_deterministic_and_upsert_idempotent(self) -> None:
        sources = [
            {"path": "a.txt", "token": "chA", "spine": True},
            {"path": "b.txt", "token": "chB", "spine": True},
        ]
        concepts = {
            0: [_concept("C/a1", 10, "chA"), _concept("C/a2", 20, "chA")],
            1: [_concept("C/b1", 10, "chB"), _concept("C/b2", 20, "chB")],
        }
        first = build._spine_stitch_edges(sources, concepts)
        second = build._spine_stitch_edges(sources, concepts)
        self.assertEqual(first, second)  # deterministic, order and shape

        # Persisting twice must not duplicate: the edge repo is keyed by the
        # synthetic (src, type, dst) id, so a re-run upserts in place.
        edges_repo = InMemoryEdgeRepository(InMemoryConceptIndexRepository())
        for edge in first:
            edges_repo.upsert(edge)
        for edge in second:
            edges_repo.upsert(edge)
        self.assertEqual(len(edges_repo.out_edges("C/a2")), 1)


class ModuleImportsOfflineTest(unittest.TestCase):
    """The module must expose its public API without importing any heavy driver."""

    def test_public_functions_are_callable(self) -> None:
        for name in ("load_manifest", "ingest", "link", "generate_questions", "status"):
            self.assertTrue(callable(getattr(build, name)), name)

    def test_package_reexports_match(self) -> None:
        import curriculum.app as app

        for name in app.__all__:
            self.assertIs(getattr(app, name), getattr(build, name))


if __name__ == "__main__":
    unittest.main()

"""Tests for the spine ingestion backbone and the edge-audit rules.

Stdlib unittest only, in the deterministic FakeLlm / in-memory style of
``tests/test_ingestion.py``. The suite pins the Task 5 behaviour contract:

* ``SpinePass`` chains the concepts of a spine-marked source into trusted
  PREREQUISITE edges (provenance "spine", confidence 1.0) in document order --
  ordered by (source order in corpus, then first source_ref line);
* a non-spine source yields no spine edges;
* ``InferEdgesPass`` stamps provenance "inferred" and caps confidence at 0.85,
  and never overwrites an existing spine edge with a duplicate PREREQUISITE;
* spine edges survive the grounding gate (VerifyPass) so the backbone actually
  reaches persistence.

Design intent under test: a wrong prerequisite edge becomes a reward bug once
unlocks are the learner-facing currency, so the backbone must come from
human-vetted ordering (a textbook's chapter sequence) while LLM inference is
restricted to lower-confidence, auditable cross-links.
"""
from __future__ import annotations

import json
import unittest

from curriculum.domain.entities import Concept, Edge, SourceRef
from curriculum.domain.enums import EdgeType
from curriculum.ingestion.passes import (
    InferEdgesPass,
    IngestionContext,
    SpinePass,
    VerifyPass,
)
from curriculum.providers_fake import FakeLlm

COURSE = "algo101"
SPINE_SOURCE = "textbook"


def _concept(cid: str, line: int, source: str = SPINE_SOURCE) -> Concept:
    """A concept grounded at ``source``:``line`` (the spine source's citation)."""
    return Concept(id=cid, course=COURSE, title=cid, source_refs=(SourceRef(source, line),))


def _ctx_with_spine(order: int = 0) -> IngestionContext:
    """Context holding three concepts A, B, C from one spine source.

    They are appended OUT of document order (C, A, B) with source_ref lines that
    encode the true order (A@10, B@20, C@30) so the test proves SpinePass sorts
    by first source_ref line rather than by insertion order.
    """
    ctx = IngestionContext(course=COURSE, spine_sources={SPINE_SOURCE})
    for cid, line in (("algo101/c", 30), ("algo101/a", 10), ("algo101/b", 20)):
        ctx.concepts.append(_concept(cid, line))
        ctx.source_of[cid] = (SPINE_SOURCE, order)
    return ctx


# --------------------------------------------------------------------------- #
# SpinePass.
# --------------------------------------------------------------------------- #
class SpinePassTest(unittest.TestCase):
    def test_spine_source_chains_consecutive_prereq_edges(self) -> None:
        ctx = _ctx_with_spine()
        SpinePass().run(ctx)
        self.assertEqual(
            [(e.src, e.dst) for e in ctx.edges],
            [("algo101/a", "algo101/b"), ("algo101/b", "algo101/c")],
        )
        for edge in ctx.edges:
            self.assertIs(edge.type, EdgeType.PREREQUISITE)
            self.assertEqual(edge.provenance, "spine")
            self.assertEqual(edge.confidence, 1.0)
            self.assertEqual(edge.rationale, f"spine order: {SPINE_SOURCE}")

    def test_spine_edges_are_grounded_at_the_source_concept(self) -> None:
        # A trusted edge must still be auditable back to a real source file, so
        # it inherits the src concept's citation.
        ctx = _ctx_with_spine()
        SpinePass().run(ctx)
        first = ctx.edges[0]
        self.assertIsNotNone(first.source_ref)
        self.assertEqual(first.source_ref.file, SPINE_SOURCE)

    def test_non_spine_source_produces_no_spine_edges(self) -> None:
        ctx = IngestionContext(course=COURSE, spine_sources={SPINE_SOURCE})
        for cid, line in (("algo101/a", 10), ("algo101/b", 20)):
            ctx.concepts.append(_concept(cid, line, source="lecture-notes"))
            ctx.source_of[cid] = ("lecture-notes", 0)
        SpinePass().run(ctx)
        self.assertEqual(ctx.edges, [])

    def test_no_spine_sources_is_a_noop(self) -> None:
        ctx = IngestionContext(course=COURSE)  # spine_sources empty by default
        for cid, line in (("algo101/a", 10), ("algo101/b", 20)):
            ctx.concepts.append(_concept(cid, line))
            ctx.source_of[cid] = (SPINE_SOURCE, 0)
        SpinePass().run(ctx)
        self.assertEqual(ctx.edges, [])

    def test_single_spine_concept_makes_no_edge(self) -> None:
        ctx = IngestionContext(course=COURSE, spine_sources={SPINE_SOURCE})
        ctx.concepts.append(_concept("algo101/only", 10))
        ctx.source_of["algo101/only"] = (SPINE_SOURCE, 0)
        SpinePass().run(ctx)
        self.assertEqual(ctx.edges, [])

    def test_spine_edges_survive_the_grounding_gate(self) -> None:
        # The whole point: a trusted backbone must reach persistence. VerifyPass
        # must NOT drop the spine edges.
        ctx = IngestionContext(
            course=COURSE,
            chunks=[{"text": "t", "file": SPINE_SOURCE, "line": 1}],
            spine_sources={SPINE_SOURCE},
        )
        for cid, line in (("algo101/a", 10), ("algo101/b", 20)):
            ctx.concepts.append(_concept(cid, line))
            ctx.source_of[cid] = (SPINE_SOURCE, 0)
        SpinePass().run(ctx)
        VerifyPass().run(ctx)
        spine_edges = [e for e in ctx.edges if e.provenance == "spine"]
        self.assertEqual([(e.src, e.dst) for e in spine_edges], [("algo101/a", "algo101/b")])


# --------------------------------------------------------------------------- #
# InferEdgesPass audit rules.
# --------------------------------------------------------------------------- #
_INFER_TRIGGER = "Infer edges for the following concepts"


def _infer_ctx() -> IngestionContext:
    ctx = IngestionContext(course=COURSE)
    for cid in ("algo101/a", "algo101/b"):
        ctx.concepts.append(Concept(id=cid, course=COURSE, title=cid))
    return ctx


class InferEdgesAuditTest(unittest.TestCase):
    def test_inferred_edges_carry_provenance_and_capped_confidence(self) -> None:
        payload = json.dumps(
            {
                "edges": [
                    {  # over-confident: must be capped to 0.85
                        "src": "algo101/a",
                        "dst": "algo101/b",
                        "type": "related",
                        "confidence": 0.99,
                        "rationale": "r",
                        "source_ref": {"file": "f.md", "line": 1},
                    },
                    {  # no confidence supplied: still <= 0.85
                        "src": "algo101/b",
                        "dst": "algo101/a",
                        "type": "prerequisite",
                        "rationale": "r",
                        "source_ref": {"file": "f.md", "line": 2},
                    },
                ]
            }
        )
        ctx = _infer_ctx()
        InferEdgesPass(FakeLlm({_INFER_TRIGGER: payload})).run(ctx)
        self.assertTrue(ctx.edges)
        for edge in ctx.edges:
            self.assertEqual(edge.provenance, "inferred")
            self.assertLessEqual(edge.confidence, 0.85)

    def test_inferred_edge_does_not_overwrite_a_spine_edge(self) -> None:
        payload = json.dumps(
            {
                "edges": [
                    {  # duplicates the spine edge: must be skipped
                        "src": "algo101/a",
                        "dst": "algo101/b",
                        "type": "prerequisite",
                        "confidence": 0.8,
                        "rationale": "llm guess",
                        "source_ref": {"file": "f.md", "line": 1},
                    },
                    {  # a different, non-conflicting relation: kept
                        "src": "algo101/a",
                        "dst": "algo101/b",
                        "type": "related",
                        "rationale": "llm cross-link",
                        "source_ref": {"file": "f.md", "line": 1},
                    },
                ]
            }
        )
        ctx = _infer_ctx()
        ctx.edges.append(
            Edge(
                src="algo101/a",
                dst="algo101/b",
                type=EdgeType.PREREQUISITE,
                rationale="spine order: textbook",
                source_ref=SourceRef(SPINE_SOURCE, 10),
                provenance="spine",
                confidence=1.0,
            )
        )
        InferEdgesPass(FakeLlm({_INFER_TRIGGER: payload})).run(ctx)

        prereq = [
            e
            for e in ctx.edges
            if e.src == "algo101/a" and e.dst == "algo101/b" and e.type is EdgeType.PREREQUISITE
        ]
        self.assertEqual(len(prereq), 1)  # the LLM duplicate was not added
        self.assertEqual(prereq[0].provenance, "spine")
        self.assertEqual(prereq[0].confidence, 1.0)
        # The non-conflicting related cross-link is still allowed through.
        related = [e for e in ctx.edges if e.type is EdgeType.RELATED]
        self.assertEqual(len(related), 1)
        self.assertEqual(related[0].provenance, "inferred")


# --------------------------------------------------------------------------- #
# Cross-source spine stitching (build orchestration).
# --------------------------------------------------------------------------- #
class SpineCrossSourceStitchTest(unittest.TestCase):
    """SpinePass chains WITHIN a source; the build stitches ACROSS sources.

    Each spine file is ingested through its own pipeline run, so SpinePass only
    ever lays intra-source edges. After the whole corpus is ingested, the build
    stitches consecutive spine sources with one ``tail(A) -> head(B)`` edge, so
    a spine split across per-chapter files is no longer disconnected.
    """

    def _run_spine_source(self, token: str, pairs) -> IngestionContext:
        ctx = IngestionContext(course=COURSE, spine_sources={token})
        for cid, line in pairs:
            ctx.concepts.append(
                Concept(id=cid, course=COURSE, title=cid, source_refs=(SourceRef(token, line),))
            )
            ctx.source_of[cid] = (token, 0)
        SpinePass().run(ctx)
        return ctx

    def test_intra_source_chains_plus_one_cross_edge(self) -> None:
        from curriculum.app import build

        ctx_a = self._run_spine_source("chA", (("algo101/a1", 10), ("algo101/a2", 20)))
        ctx_b = self._run_spine_source("chB", (("algo101/b1", 10), ("algo101/b2", 20)))

        # SpinePass laid the intra-source chains, one per file.
        self.assertEqual(
            [(e.src, e.dst) for e in ctx_a.edges], [("algo101/a1", "algo101/a2")]
        )
        self.assertEqual(
            [(e.src, e.dst) for e in ctx_b.edges], [("algo101/b1", "algo101/b2")]
        )

        sources = [
            {"path": "a.txt", "token": "chA", "spine": True},
            {"path": "b.txt", "token": "chB", "spine": True},
        ]
        cross = build._spine_stitch_edges(sources, {0: ctx_a.concepts, 1: ctx_b.concepts})

        # Exactly one cross edge, tail of A -> head of B, with the trusted shape.
        self.assertEqual([(e.src, e.dst) for e in cross], [("algo101/a2", "algo101/b1")])
        edge = cross[0]
        self.assertIs(edge.type, EdgeType.PREREQUISITE)
        self.assertEqual(edge.provenance, "spine")
        self.assertEqual(edge.confidence, 1.0)
        self.assertEqual(edge.rationale, "spine order: chA -> chB")


if __name__ == "__main__":
    unittest.main()

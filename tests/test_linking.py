"""Tests for the embedding-guided edge linker (the anti-isolation step).

Stdlib unittest only. The suite wires the real in-memory repositories
(:mod:`curriculum.storage.memory`) to a deterministic, scripted
:class:`curriculum.providers_fake.FakeLlm`, so the whole run is reproducible and
touches no network/model. The headline behaviours pinned here:

* an isolated concept (no in/out edges) gets linked to the specific neighbour the
  embeddings rank first, the edge is grounded to that concept's own source file,
  and the report counts (``isolated_before`` / ``linked`` / ``isolated_after`` /
  ``new_edges`` / ``llm_calls``) are exact;
* the empty-graph and no-isolated cases are no-ops that cost zero inference (a
  fake LLM that raises if called proves no completion happens).

Embeddings are hand-chosen basis-like vectors so that nearest-neighbour order is
unambiguous: the isolated concept ``d`` sits almost on top of ``c`` and
orthogonal to the rest, so ``nearest_to(d)`` deterministically ranks ``c`` first.
"""
from __future__ import annotations

import json
import unittest

from curriculum.domain.entities import Concept, Edge, SourceRef
from curriculum.domain.enums import EdgeType
from curriculum.linking import EmbeddingLinker
from curriculum.ports.providers import LlmProvider
from curriculum.providers_fake import FakeLlm
from curriculum.storage.memory import (
    InMemoryConceptIndexRepository,
    InMemoryEdgeRepository,
)

COURSE = "graphs"


class _NoCallLlm(LlmProvider):
    """A completion port that fails loudly if ever invoked.

    Used by the no-op tests to prove the linker makes NO paid call when there is
    nothing to link, rather than merely asserting on the returned ``llm_calls``
    count.
    """

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> str:
        raise AssertionError("the LLM must not be called when there is no work to do")


def _concept(slug: str, line: int) -> Concept:
    """Build a grounded course concept with a stable id and a single source ref."""
    cid = f"{COURSE}/{slug}"
    return Concept(
        id=cid,
        course=COURSE,
        title=slug.upper(),
        description=f"about {slug}",
        source_refs=(SourceRef("graphs.md", line),),
    )


def _repos() -> tuple[InMemoryConceptIndexRepository, InMemoryEdgeRepository]:
    concepts = InMemoryConceptIndexRepository()
    edges = InMemoryEdgeRepository(concepts)
    return concepts, edges


class LinkIsolatedTest(unittest.TestCase):
    def setUp(self) -> None:
        self.concepts, self.edges = _repos()
        # Four concepts a, b, c, d.
        for slug, line in (("a", 10), ("b", 20), ("c", 30), ("d", 40)):
            self.concepts.upsert(_concept(slug, line))
        # Distinct basis-like embeddings. d is ~colinear with c (cosine ~0.99)
        # and orthogonal to a and b, so nearest_to(d) ranks c strictly first.
        self.concepts.set_embedding(f"{COURSE}/a", [1.0, 0.0, 0.0, 0.0])
        self.concepts.set_embedding(f"{COURSE}/b", [0.0, 1.0, 0.0, 0.0])
        self.concepts.set_embedding(f"{COURSE}/c", [0.0, 0.0, 1.0, 0.0])
        self.concepts.set_embedding(f"{COURSE}/d", [0.0, 0.0, 0.9, 0.1])
        # Edges a->b and b->c leave a, b, c connected and ONLY d isolated.
        self.edges.upsert(Edge(src=f"{COURSE}/a", dst=f"{COURSE}/b", type=EdgeType.RELATED))
        self.edges.upsert(Edge(src=f"{COURSE}/b", dst=f"{COURSE}/c", type=EdgeType.RELATED))

    def _linker(self, scripts: dict[str, str]) -> EmbeddingLinker:
        return EmbeddingLinker(self.concepts, self.edges, FakeLlm(scripts))

    def test_links_isolated_to_nearest_neighbour(self) -> None:
        # Scripted answer: link d to its top-ranked neighbour c (a candidate id).
        scripts = {
            f"CONCEPT {COURSE}/d": json.dumps(
                {
                    "edges": [
                        {
                            "src": f"{COURSE}/d",
                            "dst": f"{COURSE}/c",
                            "type": "related",
                            "rationale": "d builds on c",
                        }
                    ]
                }
            )
        }
        report = self._linker(scripts).link_isolated(COURSE)

        self.assertEqual(report["isolated_before"], 1)
        self.assertEqual(report["linked"], 1)
        self.assertEqual(report["isolated_after"], 0)
        self.assertEqual(report["new_edges"], 1)
        self.assertEqual(report["llm_calls"], 1)

        created = self.edges.get(f"{COURSE}/d", f"{COURSE}/c", EdgeType.RELATED)
        self.assertIsNotNone(created)
        self.assertEqual(created.rationale, "d builds on c")
        # Grounded to the isolated endpoint's (d's) own source file.
        self.assertIsNotNone(created.source_ref)
        self.assertEqual(created.source_ref.file, "graphs.md")
        self.assertEqual(created.source_ref.line, 40)
        # d is no longer isolated.
        self.assertTrue(self.edges.out_edges(f"{COURSE}/d"))

    def test_nearest_to_ranks_the_expected_neighbour_first(self) -> None:
        # Guards the embedding setup the link test relies on: c must rank first.
        nearest = self.concepts.nearest_to(f"{COURSE}/d", course=COURSE, k=10)
        self.assertEqual(nearest[0][0], f"{COURSE}/c")

    def test_edge_to_unknown_endpoint_is_rejected(self) -> None:
        # The model invents an endpoint that is neither a candidate nor a known
        # concept; the linker must drop it and leave d isolated.
        scripts = {
            f"CONCEPT {COURSE}/d": json.dumps(
                {"edges": [{"src": f"{COURSE}/d", "dst": f"{COURSE}/ghost", "type": "related"}]}
            )
        }
        report = self._linker(scripts).link_isolated(COURSE)
        self.assertEqual(report["new_edges"], 0)
        self.assertEqual(report["linked"], 0)
        self.assertEqual(report["isolated_after"], 1)
        # One call was still made (one group), but nothing was kept.
        self.assertEqual(report["llm_calls"], 1)
        self.assertIsNone(self.edges.get(f"{COURSE}/d", f"{COURSE}/ghost", EdgeType.RELATED))

    def test_self_loop_is_rejected(self) -> None:
        scripts = {
            f"CONCEPT {COURSE}/d": json.dumps(
                {"edges": [{"src": f"{COURSE}/d", "dst": f"{COURSE}/d", "type": "related"}]}
            )
        }
        report = self._linker(scripts).link_isolated(COURSE)
        self.assertEqual(report["new_edges"], 0)
        self.assertEqual(report["isolated_after"], 1)

    def test_run_is_deterministic(self) -> None:
        scripts = {
            f"CONCEPT {COURSE}/d": json.dumps(
                {"edges": [{"src": f"{COURSE}/d", "dst": f"{COURSE}/c", "type": "related"}]}
            )
        }
        first = dict(self._linker(scripts).link_isolated(COURSE))
        # Re-run against fresh repos seeded identically: same report exactly.
        self.setUp()
        second = dict(self._linker(scripts).link_isolated(COURSE))
        self.assertEqual(first, second)


class NoOpTest(unittest.TestCase):
    def test_empty_graph_is_a_noop_without_calling_the_llm(self) -> None:
        concepts, edges = _repos()
        linker = EmbeddingLinker(concepts, edges, _NoCallLlm())
        report = linker.link_isolated(COURSE)
        self.assertEqual(
            dict(report),
            {
                "isolated_before": 0,
                "linked": 0,
                "isolated_after": 0,
                "new_edges": 0,
                "llm_calls": 0,
            },
        )

    def test_no_isolated_concepts_is_a_noop_without_calling_the_llm(self) -> None:
        concepts, edges = _repos()
        concepts.upsert(_concept("a", 10))
        concepts.upsert(_concept("b", 20))
        # Both concepts are connected, so there is nothing isolated to repair.
        edges.upsert(Edge(src=f"{COURSE}/a", dst=f"{COURSE}/b", type=EdgeType.RELATED))
        report = EmbeddingLinker(concepts, edges, _NoCallLlm()).link_isolated(COURSE)
        self.assertEqual(report["isolated_before"], 0)
        self.assertEqual(report["new_edges"], 0)
        self.assertEqual(report["llm_calls"], 0)


if __name__ == "__main__":
    unittest.main()

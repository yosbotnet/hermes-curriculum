"""Tests for the multipass ingestion pipeline.

Stdlib unittest only. The suite drives the standard ``default_pipeline`` with
the deterministic ``FakeLlm`` (scripted JSON) + ``FakeEmbedder`` and the
in-memory repositories, then asserts the headline behaviours:

* concepts, edges and questions persist through the repository ports;
* ``DedupePass`` merges two near-identical concepts (and unions their refs);
* ``VerifyPass`` drops an ungrounded concept and cascades the drop to its
  dangling edges and orphan questions.

Focused unit tests pin each pass's contract (JSON parsing, type mapping,
difficulty/hop grading, the grounding gate) so a regression localises cleanly.
"""
from __future__ import annotations

import json
import unittest

from curriculum.domain.entities import (
    Concept,
    ConceptContent,
    Edge,
    Question,
    QuestionContent,
    SourceRef,
)
from curriculum.domain.enums import EdgeType
from curriculum.ingestion.passes import (
    DedupePass,
    ExtractPass,
    InferEdgesPass,
    IngestionContext,
    IngestionPass,
    QuestionGenPass,
    VerifyPass,
)
from curriculum.ingestion.pipeline import Pipeline, default_pipeline
from curriculum.providers_fake import FakeEmbedder, FakeLlm
from curriculum.storage.memory import (
    InMemoryConceptIndexRepository,
    InMemoryContentRepository,
    InMemoryEdgeRepository,
    InMemoryQuestionRepository,
)

COURSE = "algo101"

# Shared body so Dijkstra and its duplicate embed to the identical vector and so
# DedupePass (FakeEmbedder is hash-deterministic: identical text -> identical
# vector -> cosine 1.0) folds them together.
_DIJKSTRA_BODY = "Dijkstra computes shortest paths from a source."

CHUNKS = [
    {
        "text": "Dijkstra's algorithm computes shortest paths from a single source vertex.",
        "file": "algorithms.md",
        "line": 10,
    },
    {
        "text": "Edge relaxation updates the tentative distance of each neighbour.",
        "file": "algorithms.md",
        "line": 25,
    },
    {
        "text": "A priority queue orders the search frontier by key.",
        "file": "ds.md",
        "line": 5,
    },
]

_RELAX_EDGE_ID = "algo101/relaxation::prerequisite::algo101/dijkstra"
_GHOST_EDGE_ID = "algo101/ghost::related::algo101/dijkstra"


def _scripts() -> dict[str, str]:
    """Scripted FakeLlm answers keyed on substrings unique to each prompt.

    The fake returns the value whose trigger is the longest substring present
    in the prompt; the triggers below are chosen so exactly one matches each
    pass's prompt (chunk text for extraction, the concept/edge id for question
    generation, a stable header for edge inference).
    """
    return {
        # --- extraction: one trigger per chunk's verbatim text -------------- #
        "Dijkstra's algorithm computes shortest paths": json.dumps(
            {
                "concepts": [
                    {
                        "id": "algo101/dijkstra",
                        "title": "Dijkstra",
                        "description": "Single-source shortest path procedure",
                        "body": _DIJKSTRA_BODY,
                        "importance": 0.9,
                        "source_refs": [{"file": "algorithms.md", "line": 10}],
                    },
                    {
                        # Near-identical duplicate (same body) -> merged by dedupe.
                        "id": "algo101/dijkstra-dup",
                        "title": "Dijkstra Algorithm",
                        "description": "Shortest path algorithm restated",
                        "body": _DIJKSTRA_BODY,
                        "importance": 0.8,
                        "source_refs": [{"file": "algorithms.md", "line": 11}],
                    },
                    {
                        # Ungrounded (empty refs) -> dropped by VerifyPass.
                        "id": "algo101/ghost",
                        "title": "Ghost",
                        "description": "Unsupported claim",
                        "body": "An unsourced fabricated claim.",
                        "importance": 0.5,
                        "source_refs": [],
                    },
                ]
            }
        ),
        "Edge relaxation updates the tentative distance": json.dumps(
            {
                "concepts": [
                    {
                        "id": "algo101/relaxation",
                        "title": "Relaxation",
                        "description": "Tentative distance update step",
                        "body": "Relaxation lowers a neighbour distance estimate.",
                        "importance": 0.7,
                        "source_refs": [{"file": "algorithms.md", "line": 25}],
                    }
                ]
            }
        ),
        "priority queue orders the search frontier": json.dumps(
            {
                "concepts": [
                    {
                        "id": "algo101/priority-queue",
                        "title": "Priority Queue",
                        "description": "Min-key frontier structure",
                        "body": "A priority queue yields the minimum-key element.",
                        "importance": 0.6,
                        "source_refs": [{"file": "ds.md", "line": 5}],
                    }
                ]
            }
        ),
        # --- edge inference ------------------------------------------------- #
        "Infer edges for the following concepts": json.dumps(
            {
                "edges": [
                    {
                        "src": "algo101/relaxation",
                        "dst": "algo101/dijkstra",
                        "type": "prerequisite",
                        "weight": 1.0,
                        "importance": 0.8,
                        "rationale": "Relaxation underpins Dijkstra",
                        "source_ref": {"file": "algorithms.md", "line": 25},
                    },
                    {
                        "src": "algo101/priority-queue",
                        "dst": "algo101/dijkstra",
                        "type": "prerequisite",
                        "weight": 1.0,
                        "importance": 0.3,
                        "rationale": "PQ used by Dijkstra",
                        "source_ref": {"file": "ds.md", "line": 5},
                    },
                    {
                        # References the ghost concept -> pruned (dangling) by verify.
                        "src": "algo101/ghost",
                        "dst": "algo101/dijkstra",
                        "type": "related",
                        "weight": 0.5,
                        "importance": 0.9,
                        "rationale": "ghostly link",
                        "source_ref": {"file": "algorithms.md", "line": 12},
                    },
                ]
            }
        ),
        # --- question generation: per concept ------------------------------- #
        "Generate exam questions for concept algo101/dijkstra": json.dumps(
            {
                "questions": [
                    {
                        "kind": "open",
                        "difficulty": 3,
                        "prompt": "Explain how Dijkstra finds shortest paths.",
                        "rubric": "Mentions relaxation and priority queue.",
                        "source_refs": [{"file": "algorithms.md", "line": 10}],
                    }
                ]
            }
        ),
        "Generate exam questions for concept algo101/ghost": json.dumps(
            {"questions": [{"kind": "open", "difficulty": 1, "prompt": "Ghost question."}]}
        ),
        "Generate exam questions for concept algo101/relaxation": json.dumps(
            {
                "questions": [
                    {
                        "kind": "open",
                        "difficulty": 2,
                        "prompt": "What does relaxation do?",
                        "rubric": "Distance estimate decreases.",
                    }
                ]
            }
        ),
        "Generate exam questions for concept algo101/priority-queue": json.dumps(
            {
                "questions": [
                    {
                        "kind": "mcq",
                        "difficulty": 2,
                        "prompt": "What does a priority queue return?",
                        "rubric": "The minimum-key element.",
                    }
                ]
            }
        ),
        # --- question generation: per important edge ------------------------ #
        f"Generate a multi-hop question for edge {_RELAX_EDGE_ID}": json.dumps(
            {
                "questions": [
                    {
                        "kind": "open",
                        "difficulty": 4,
                        "hop_count": 2,
                        "prompt": "How does relaxation enable Dijkstra?",
                        "rubric": "Connects both concepts.",
                    }
                ]
            }
        ),
        f"Generate a multi-hop question for edge {_GHOST_EDGE_ID}": json.dumps(
            {"questions": [{"kind": "open", "prompt": "Ghost edge question."}]}
        ),
    }


def _repos() -> tuple[
    InMemoryConceptIndexRepository,
    InMemoryEdgeRepository,
    InMemoryQuestionRepository,
    InMemoryContentRepository,
]:
    concepts = InMemoryConceptIndexRepository()
    edges = InMemoryEdgeRepository(concepts)
    questions = InMemoryQuestionRepository()
    content = InMemoryContentRepository()
    return concepts, edges, questions, content


# --------------------------------------------------------------------------- #
# Full pipeline integration.
# --------------------------------------------------------------------------- #
class FullPipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.embedder = FakeEmbedder(dim=64)
        self.llm = FakeLlm(_scripts())
        self.pipeline = default_pipeline(self.llm, self.embedder)
        self.ctx = IngestionContext(course=COURSE, chunks=list(CHUNKS))
        self.concepts, self.edges, self.questions, self.content = _repos()

    def _run_and_persist(self) -> dict:
        self.pipeline.run(self.ctx)
        return dict(
            self.pipeline.persist(
                self.ctx,
                concepts=self.concepts,
                edges=self.edges,
                questions=self.questions,
                content=self.content,
            )
        )

    def test_persist_counts(self) -> None:
        counts = self._run_and_persist()
        # dijkstra (+merged dup), relaxation, priority-queue survive; ghost dropped.
        self.assertEqual(counts["concepts"], 3)
        # ghost edge pruned (dangling); relaxation+pq prerequisite edges remain.
        self.assertEqual(counts["edges"], 2)
        # 3 concept questions + 1 important-edge question survive verify.
        self.assertEqual(counts["questions"], 4)

    def test_concepts_persisted_and_ghost_dropped(self) -> None:
        self._run_and_persist()
        self.assertIsNotNone(self.concepts.get("algo101/dijkstra"))
        self.assertIsNotNone(self.concepts.get("algo101/relaxation"))
        self.assertIsNotNone(self.concepts.get("algo101/priority-queue"))
        # The ungrounded concept is gone (VerifyPass grounding gate).
        self.assertIsNone(self.concepts.get("algo101/ghost"))
        # The merged duplicate never reaches storage under its own id.
        self.assertIsNone(self.concepts.get("algo101/dijkstra-dup"))

    def test_dedupe_unions_source_refs_onto_survivor(self) -> None:
        self._run_and_persist()
        dijkstra = self.concepts.get("algo101/dijkstra")
        # lines 10 (original) and 11 (from the merged duplicate) both kept.
        files_lines = sorted((r.file, r.line) for r in dijkstra.source_refs)
        self.assertEqual(files_lines, [("algorithms.md", 10), ("algorithms.md", 11)])

    def test_concept_content_persisted_and_hash_linked(self) -> None:
        self._run_and_persist()
        stored = self.concepts.get("algo101/dijkstra")
        self.assertIsNotNone(stored.content_hash)
        content = self.content.get_concept_content("algo101/dijkstra")
        self.assertIsNotNone(content)
        self.assertEqual(content.body, _DIJKSTRA_BODY)

    def test_edges_persisted_with_metadata(self) -> None:
        self._run_and_persist()
        relax = self.edges.get("algo101/relaxation", "algo101/dijkstra", EdgeType.PREREQUISITE)
        self.assertIsNotNone(relax)
        self.assertEqual(relax.rationale, "Relaxation underpins Dijkstra")
        self.assertIsNotNone(relax.source_ref)
        # The dangling ghost edge was pruned.
        self.assertIsNone(self.edges.get("algo101/ghost", "algo101/dijkstra", EdgeType.RELATED))
        self.assertEqual(len(self.edges.list_by_course(COURSE)), 2)

    def test_concept_question_persisted(self) -> None:
        self._run_and_persist()
        qs = self.questions.by_concept("algo101/dijkstra")
        self.assertEqual(len(qs), 1)
        self.assertEqual(qs[0].difficulty, 3)
        self.assertEqual(qs[0].hop_count, 1)
        self.assertIsNone(qs[0].edge_id)
        qc = self.content.get_question_content(qs[0].id)
        self.assertIsNotNone(qc)
        self.assertIn("Dijkstra", qc.prompt)

    def test_edge_question_is_multi_hop_and_tagged(self) -> None:
        self._run_and_persist()
        edge_qs = self.questions.by_edge(_RELAX_EDGE_ID)
        self.assertEqual(len(edge_qs), 1)
        self.assertEqual(edge_qs[0].hop_count, 2)
        self.assertEqual(edge_qs[0].difficulty, 4)
        self.assertEqual(edge_qs[0].edge_id, _RELAX_EDGE_ID)

    def test_unimportant_edge_gets_no_question(self) -> None:
        self._run_and_persist()
        # pq->dijkstra has importance 0.3 (< 0.5 default) -> no question minted.
        self.assertEqual(
            self.questions.by_edge("algo101/priority-queue::prerequisite::algo101/dijkstra"),
            [],
        )

    def test_orphan_questions_dropped(self) -> None:
        self._run_and_persist()
        # Ghost concept question and ghost edge question never persist.
        self.assertIsNone(self.questions.get("algo101/ghost::q0"))
        self.assertIsNone(self.questions.get(f"{_GHOST_EDGE_ID}::q0"))

    def test_embedding_cache_enables_nearest(self) -> None:
        self._run_and_persist()
        query = self.embedder.embed([_DIJKSTRA_BODY])[0]
        nearest = self.concepts.nearest(query, course=COURSE, k=1)
        self.assertEqual(nearest[0][0], "algo101/dijkstra")

    def test_run_is_deterministic(self) -> None:
        first = self._run_and_persist()
        # A fresh run with fresh repos must reproduce the same counts exactly.
        ctx2 = IngestionContext(course=COURSE, chunks=list(CHUNKS))
        repos2 = _repos()
        default_pipeline(self.llm, self.embedder).run(ctx2)
        second = dict(
            default_pipeline(self.llm, self.embedder).persist(
                ctx2,
                concepts=repos2[0],
                edges=repos2[1],
                questions=repos2[2],
                content=repos2[3],
            )
        )
        self.assertEqual(first, second)


# --------------------------------------------------------------------------- #
# ExtractPass.
# --------------------------------------------------------------------------- #
class ExtractPassTest(unittest.TestCase):
    def test_builds_concept_and_content_with_refs(self) -> None:
        llm = FakeLlm(_scripts())
        ctx = IngestionContext(course=COURSE, chunks=[CHUNKS[1]])
        ExtractPass(llm).run(ctx)
        self.assertEqual([c.id for c in ctx.concepts], ["algo101/relaxation"])
        concept = ctx.concepts[0]
        self.assertEqual(concept.course, COURSE)
        self.assertEqual(concept.source_refs, (SourceRef("algorithms.md", 25),))
        content = ctx.concept_content_for("algo101/relaxation")
        self.assertIsNotNone(content)
        self.assertTrue(content.body)

    def test_unmatched_prompt_yields_no_concepts(self) -> None:
        # No script matches -> FakeLlm default stub is not JSON -> no concepts,
        # and crucially no exception (best-effort ingestion).
        ctx = IngestionContext(
            course=COURSE, chunks=[{"text": "unrelated", "file": "x.md", "line": 1}]
        )
        ExtractPass(FakeLlm()).run(ctx)
        self.assertEqual(ctx.concepts, [])

    def test_missing_id_falls_back_to_title_slug(self) -> None:
        payload = json.dumps(
            {"concepts": [{"title": "Big O Notation", "body": "growth", "source_refs": []}]}
        )
        llm = FakeLlm({"slugme": payload})
        ctx = IngestionContext(
            course=COURSE, chunks=[{"text": "slugme", "file": "x.md", "line": 1}]
        )
        ExtractPass(llm).run(ctx)
        self.assertEqual(ctx.concepts[0].id, "algo101/big-o-notation")

    def test_prose_wrapped_json_is_parsed(self) -> None:
        payload = (
            "Sure, here you go:\n```json\n"
            + json.dumps({"concepts": [{"id": "algo101/x", "title": "X", "body": "b",
                                        "source_refs": [{"file": "x.md", "line": 2}]}]})
            + "\n```\nHope that helps!"
        )
        llm = FakeLlm({"wrapme": payload})
        ctx = IngestionContext(
            course=COURSE, chunks=[{"text": "wrapme", "file": "x.md", "line": 1}]
        )
        ExtractPass(llm).run(ctx)
        self.assertEqual([c.id for c in ctx.concepts], ["algo101/x"])


# --------------------------------------------------------------------------- #
# DedupePass.
# --------------------------------------------------------------------------- #
class DedupePassTest(unittest.TestCase):
    def _ctx_with(self, *concepts: tuple[str, str, tuple[SourceRef, ...]]) -> IngestionContext:
        ctx = IngestionContext(course=COURSE)
        for cid, body, refs in concepts:
            ctx.concepts.append(Concept(id=cid, course=COURSE, title=cid, source_refs=refs))
            ctx.concept_content.append(
                ConceptContent(concept_id=cid, title=cid, body=body, source_refs=refs)
            )
        return ctx

    def test_merges_near_identical_concepts(self) -> None:
        ctx = self._ctx_with(
            ("a", _DIJKSTRA_BODY, (SourceRef("f.md", 1),)),
            ("b", _DIJKSTRA_BODY, (SourceRef("f.md", 2),)),
        )
        DedupePass(FakeEmbedder(dim=64)).run(ctx)
        self.assertEqual([c.id for c in ctx.concepts], ["a"])
        # Survivor inherits the duplicate's provenance.
        self.assertEqual(
            sorted((r.file, r.line) for r in ctx.concepts[0].source_refs),
            [("f.md", 1), ("f.md", 2)],
        )
        # Content list collapses in lockstep.
        self.assertEqual([cc.concept_id for cc in ctx.concept_content], ["a"])

    def test_distinct_concepts_are_not_merged(self) -> None:
        ctx = self._ctx_with(
            ("a", "graphs and shortest paths", (SourceRef("f.md", 1),)),
            ("b", "hash tables and probing", (SourceRef("f.md", 2),)),
        )
        DedupePass(FakeEmbedder(dim=64)).run(ctx)
        self.assertEqual({c.id for c in ctx.concepts}, {"a", "b"})

    def test_records_embedding_for_survivor(self) -> None:
        ctx = self._ctx_with(("a", "unique body", (SourceRef("f.md", 1),)))
        DedupePass(FakeEmbedder(dim=64)).run(ctx)
        self.assertIn("a", ctx.embeddings)
        self.assertEqual(len(ctx.embeddings["a"]), 64)

    def test_empty_context_is_noop(self) -> None:
        ctx = IngestionContext(course=COURSE)
        DedupePass(FakeEmbedder()).run(ctx)
        self.assertEqual(ctx.concepts, [])


# --------------------------------------------------------------------------- #
# InferEdgesPass.
# --------------------------------------------------------------------------- #
class InferEdgesPassTest(unittest.TestCase):
    def _ctx(self) -> IngestionContext:
        ctx = IngestionContext(course=COURSE)
        for cid in ("algo101/dijkstra", "algo101/relaxation", "algo101/priority-queue", "algo101/ghost"):
            ctx.concepts.append(Concept(id=cid, course=COURSE, title=cid))
        return ctx

    def test_maps_types_and_builds_edges(self) -> None:
        ctx = self._ctx()
        InferEdgesPass(FakeLlm(_scripts())).run(ctx)
        by_id = {e.id: e for e in ctx.edges}
        self.assertIn(_RELAX_EDGE_ID, by_id)
        self.assertIs(by_id[_RELAX_EDGE_ID].type, EdgeType.PREREQUISITE)
        self.assertEqual(by_id[_RELAX_EDGE_ID].importance, 0.8)

    def test_skips_unknown_type_and_self_loop(self) -> None:
        payload = json.dumps(
            {
                "edges": [
                    {"src": "a", "dst": "b", "type": "bogus", "source_ref": {"file": "f.md"}},
                    {"src": "a", "dst": "a", "type": "related", "source_ref": {"file": "f.md"}},
                    {"src": "a", "dst": "b", "type": "related", "source_ref": {"file": "f.md"}},
                ]
            }
        )
        ctx = IngestionContext(course=COURSE)
        ctx.concepts.append(Concept(id="a", course=COURSE, title="A"))
        ctx.concepts.append(Concept(id="b", course=COURSE, title="B"))
        InferEdgesPass(FakeLlm({"Infer edges for the following concepts": payload})).run(ctx)
        self.assertEqual([(e.src, e.dst) for e in ctx.edges], [("a", "b")])

    def test_no_concepts_means_no_llm_call(self) -> None:
        ctx = IngestionContext(course=COURSE)
        InferEdgesPass(FakeLlm(_scripts())).run(ctx)
        self.assertEqual(ctx.edges, [])


# --------------------------------------------------------------------------- #
# QuestionGenPass.
# --------------------------------------------------------------------------- #
class QuestionGenPassTest(unittest.TestCase):
    def test_grades_difficulty_and_hop_for_concept_and_edge(self) -> None:
        ctx = IngestionContext(course=COURSE)
        ctx.concepts.append(
            Concept(id="algo101/dijkstra", course=COURSE, title="Dijkstra",
                    source_refs=(SourceRef("algorithms.md", 10),))
        )
        ctx.concept_content.append(
            ConceptContent(concept_id="algo101/dijkstra", title="Dijkstra", body=_DIJKSTRA_BODY)
        )
        ctx.concepts.append(
            Concept(id="algo101/relaxation", course=COURSE, title="Relaxation",
                    source_refs=(SourceRef("algorithms.md", 25),))
        )
        ctx.edges.append(
            Edge(src="algo101/relaxation", dst="algo101/dijkstra", type=EdgeType.PREREQUISITE,
                 importance=0.8, rationale="r", source_ref=SourceRef("algorithms.md", 25))
        )
        QuestionGenPass(FakeLlm(_scripts())).run(ctx)
        concept_q = next(q for q in ctx.questions if q.concept_id == "algo101/dijkstra"
                         and q.edge_id is None)
        self.assertEqual(concept_q.difficulty, 3)
        self.assertEqual(concept_q.hop_count, 1)
        edge_q = next(q for q in ctx.questions if q.edge_id == _RELAX_EDGE_ID)
        self.assertEqual(edge_q.hop_count, 2)
        self.assertEqual(edge_q.difficulty, 4)
        self.assertEqual(edge_q.generated_by, "ingestion.QuestionGenPass")

    def test_importance_threshold_filters_edges(self) -> None:
        ctx = IngestionContext(course=COURSE)
        low = Edge(src="x", dst="y", type=EdgeType.RELATED, importance=0.4,
                   source_ref=SourceRef("f.md", 1))
        ctx.edges.append(low)
        # The edge prompt would match no script; even so it must not be called
        # because importance 0.4 < 0.5 short-circuits before any LLM call.
        QuestionGenPass(FakeLlm(_scripts())).run(ctx)
        self.assertEqual(ctx.questions, [])

    def test_difficulty_is_clamped(self) -> None:
        payload = json.dumps({"questions": [{"prompt": "q", "difficulty": 99}]})
        ctx = IngestionContext(course=COURSE)
        ctx.concepts.append(Concept(id="c", course=COURSE, title="C",
                                    source_refs=(SourceRef("f.md", 1),)))
        QuestionGenPass(
            FakeLlm({"Generate exam questions for concept c": payload})
        ).run(ctx)
        self.assertEqual(ctx.questions[0].difficulty, 5)


# --------------------------------------------------------------------------- #
# VerifyPass.
# --------------------------------------------------------------------------- #
class VerifyPassTest(unittest.TestCase):
    def _ctx(self) -> IngestionContext:
        return IngestionContext(
            course=COURSE,
            chunks=[{"text": "t", "file": "real.md", "line": 1}],
        )

    def test_drops_empty_ref_concept(self) -> None:
        ctx = self._ctx()
        ctx.concepts.append(Concept(id="grounded", course=COURSE, title="G",
                                    source_refs=(SourceRef("real.md", 1),)))
        ctx.concepts.append(Concept(id="empty", course=COURSE, title="E", source_refs=()))
        VerifyPass().run(ctx)
        self.assertEqual([c.id for c in ctx.concepts], ["grounded"])

    def test_drops_hallucinated_file_concept(self) -> None:
        ctx = self._ctx()
        ctx.concepts.append(
            Concept(id="fake", course=COURSE, title="F",
                    source_refs=(SourceRef("does-not-exist.md", 1),))
        )
        VerifyPass().run(ctx)
        self.assertEqual(ctx.concepts, [])

    def test_drops_dangling_and_uncited_edges(self) -> None:
        ctx = self._ctx()
        ctx.concepts.append(Concept(id="a", course=COURSE, title="A",
                                    source_refs=(SourceRef("real.md", 1),)))
        ctx.concepts.append(Concept(id="b", course=COURSE, title="B",
                                    source_refs=(SourceRef("real.md", 2),)))
        good = Edge(src="a", dst="b", type=EdgeType.RELATED,
                    source_ref=SourceRef("real.md", 3))
        dangling = Edge(src="a", dst="ghost", type=EdgeType.RELATED,
                        source_ref=SourceRef("real.md", 4))
        uncited = Edge(src="a", dst="b", type=EdgeType.PREREQUISITE, source_ref=None)
        ctx.edges.extend([good, dangling, uncited])
        VerifyPass().run(ctx)
        self.assertEqual([(e.src, e.dst, e.type) for e in ctx.edges],
                         [("a", "b", EdgeType.RELATED)])

    def test_cascades_drop_to_orphan_questions(self) -> None:
        ctx = self._ctx()
        ctx.concepts.append(Concept(id="a", course=COURSE, title="A",
                                    source_refs=(SourceRef("real.md", 1),)))
        # Question on a concept that will be dropped (no grounded concept "gone").
        ctx.questions.append(Question(id="gone::q0", concept_id="gone"))
        ctx.question_content.append(QuestionContent(question_id="gone::q0", prompt="p"))
        ctx.questions.append(Question(id="a::q0", concept_id="a",
                                      source_refs=(SourceRef("real.md", 1),)))
        ctx.question_content.append(QuestionContent(question_id="a::q0", prompt="p"))
        VerifyPass().run(ctx)
        self.assertEqual([q.id for q in ctx.questions], ["a::q0"])
        self.assertEqual([qc.question_id for qc in ctx.question_content], ["a::q0"])


# --------------------------------------------------------------------------- #
# Pipeline assembly.
# --------------------------------------------------------------------------- #
class PipelineAssemblyTest(unittest.TestCase):
    def test_default_pipeline_order_and_types(self) -> None:
        pipeline = default_pipeline(FakeLlm(), FakeEmbedder())
        self.assertEqual(
            [type(p) for p in pipeline.passes],
            [ExtractPass, DedupePass, InferEdgesPass, QuestionGenPass, VerifyPass],
        )

    def test_all_passes_are_ingestion_passes(self) -> None:
        for p in default_pipeline(FakeLlm(), FakeEmbedder()).passes:
            self.assertIsInstance(p, IngestionPass)

    def test_passes_run_in_given_order(self) -> None:
        calls: list[str] = []

        class Recorder(IngestionPass):
            def __init__(self, tag: str) -> None:
                self.tag = tag

            def run(self, ctx: IngestionContext) -> None:
                calls.append(self.tag)

        Pipeline([Recorder("one"), Recorder("two"), Recorder("three")]).run(
            IngestionContext(course=COURSE)
        )
        self.assertEqual(calls, ["one", "two", "three"])

    def test_passes_view_is_immutable_snapshot(self) -> None:
        passes = [ExtractPass(FakeLlm())]
        pipeline = Pipeline(passes)
        passes.append(VerifyPass())  # mutate the caller's list after construction
        self.assertEqual(len(pipeline.passes), 1)


if __name__ == "__main__":
    unittest.main()

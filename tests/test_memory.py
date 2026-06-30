"""Tests for the in-memory repository adapters.

Stdlib unittest only. The adapters are pure and deterministic, so every
assertion is reproducible without an injected clock beyond the fixed datetimes
we build here. We exercise upsert/get round-trips for every repo plus the
non-trivial invariants: cosine ``nearest`` ordering, type-filtered edge
traversal, exposure/skip accounting, stable content hashing, and course
resolution for the course-less ``Edge``/``LearnerState`` rows.
"""
from __future__ import annotations

import unittest
from datetime import datetime, timedelta

from curriculum.domain.entities import (
    Concept,
    ConceptContent,
    CourseProfile,
    Edge,
    LearnerState,
    Question,
    QuestionContent,
    ReviewEvent,
    SourceRef,
)
from curriculum.domain.enums import EdgeType, FsrsRating
from curriculum.storage.memory import (
    InMemoryConceptIndexRepository,
    InMemoryContentRepository,
    InMemoryCourseProfileRepository,
    InMemoryEdgeRepository,
    InMemoryLearnerStateRepository,
    InMemoryQuestionRepository,
    InMemoryReviewLogRepository,
)

T0 = datetime(2026, 1, 1, 12, 0, 0)


def _concept(cid: str, course: str = "cs101", **kw) -> Concept:
    title = kw.pop("title", cid.title())
    return Concept(id=cid, course=course, title=title, **kw)


class ConceptIndexTest(unittest.TestCase):
    def setUp(self) -> None:
        self.repo = InMemoryConceptIndexRepository()

    def test_upsert_get_round_trip(self) -> None:
        c = _concept("a")
        self.repo.upsert(c)
        self.assertEqual(self.repo.get("a"), c)

    def test_get_missing_returns_none(self) -> None:
        self.assertIsNone(self.repo.get("nope"))

    def test_upsert_overwrites_in_place(self) -> None:
        self.repo.upsert(_concept("a", title="Old"))
        self.repo.upsert(Concept(id="a", course="cs101", title="New"))
        self.assertEqual(self.repo.get("a").title, "New")

    def test_list_by_course_filters_and_sorts(self) -> None:
        self.repo.upsert(_concept("b", course="cs101"))
        self.repo.upsert(_concept("a", course="cs101"))
        self.repo.upsert(_concept("z", course="other"))
        listed = self.repo.list_by_course("cs101")
        self.assertEqual([c.id for c in listed], ["a", "b"])

    def test_delete_removes_concept_and_embedding(self) -> None:
        self.repo.upsert(_concept("a"))
        self.repo.set_embedding("a", [1.0, 0.0])
        self.repo.delete("a")
        self.assertIsNone(self.repo.get("a"))
        # With its concept gone, the embedding can no longer match a course.
        self.assertEqual(self.repo.nearest([1.0, 0.0], course="cs101"), [])

    def test_delete_missing_is_noop(self) -> None:
        self.repo.delete("ghost")  # must not raise

    def test_set_embedding_snapshots_vector(self) -> None:
        self.repo.upsert(_concept("a"))
        vec = [1.0, 0.0, 0.0]
        self.repo.set_embedding("a", vec)
        vec[0] = -99.0  # mutate caller's list after the fact
        # Stored copy must be unaffected: a should still match [1,0,0] perfectly.
        result = self.repo.nearest([1.0, 0.0, 0.0], course="cs101", k=1)
        self.assertAlmostEqual(result[0][1], 1.0)

    def test_nearest_returns_closest_first(self) -> None:
        for cid in ("a", "b", "c"):
            self.repo.upsert(_concept(cid))
        self.repo.set_embedding("a", [1.0, 0.0])      # cosine 1.0 with query
        self.repo.set_embedding("b", [0.0, 1.0])      # cosine 0.0
        self.repo.set_embedding("c", [0.9, 0.1])      # close to query
        result = self.repo.nearest([1.0, 0.0], course="cs101", k=3)
        ids = [cid for cid, _ in result]
        self.assertEqual(ids[0], "a")
        self.assertEqual(ids[1], "c")
        self.assertEqual(ids[2], "b")
        # Similarities are descending.
        sims = [s for _, s in result]
        self.assertEqual(sims, sorted(sims, reverse=True))

    def test_nearest_filters_by_course(self) -> None:
        self.repo.upsert(_concept("a", course="cs101"))
        self.repo.upsert(_concept("x", course="other"))
        self.repo.set_embedding("a", [1.0, 0.0])
        self.repo.set_embedding("x", [1.0, 0.0])
        result = self.repo.nearest([1.0, 0.0], course="cs101")
        self.assertEqual([cid for cid, _ in result], ["a"])

    def test_nearest_honours_k(self) -> None:
        for cid in ("a", "b", "c"):
            self.repo.upsert(_concept(cid))
            self.repo.set_embedding(cid, [1.0, 0.0])
        self.assertEqual(len(self.repo.nearest([1.0, 0.0], course="cs101", k=2)), 2)

    def test_nearest_non_positive_k_is_empty(self) -> None:
        self.repo.upsert(_concept("a"))
        self.repo.set_embedding("a", [1.0, 0.0])
        self.assertEqual(self.repo.nearest([1.0, 0.0], course="cs101", k=0), [])

    def test_nearest_zero_vector_similarity_is_zero(self) -> None:
        self.repo.upsert(_concept("a"))
        self.repo.set_embedding("a", [0.0, 0.0])
        result = self.repo.nearest([1.0, 0.0], course="cs101", k=1)
        self.assertEqual(result[0][1], 0.0)

    def test_nearest_tie_breaks_by_id(self) -> None:
        for cid in ("b", "a"):
            self.repo.upsert(_concept(cid))
            self.repo.set_embedding(cid, [1.0, 0.0])
        result = self.repo.nearest([1.0, 0.0], course="cs101", k=2)
        self.assertEqual([cid for cid, _ in result], ["a", "b"])


class EdgeRepositoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.concepts = InMemoryConceptIndexRepository()
        self.concepts.upsert(_concept("a"))
        self.concepts.upsert(_concept("b"))
        self.concepts.upsert(_concept("c"))
        self.repo = InMemoryEdgeRepository(self.concepts)

    def test_upsert_get_round_trip(self) -> None:
        e = Edge(src="a", dst="b", type=EdgeType.PREREQUISITE)
        self.repo.upsert(e)
        self.assertEqual(self.repo.get("a", "b", EdgeType.PREREQUISITE), e)

    def test_get_missing_returns_none(self) -> None:
        self.assertIsNone(self.repo.get("a", "b", EdgeType.RELATED))

    def test_get_discriminates_by_type(self) -> None:
        self.repo.upsert(Edge(src="a", dst="b", type=EdgeType.PREREQUISITE))
        self.assertIsNone(self.repo.get("a", "b", EdgeType.RELATED))

    def test_out_edges_filter_by_type(self) -> None:
        self.repo.upsert(Edge(src="a", dst="b", type=EdgeType.PREREQUISITE))
        self.repo.upsert(Edge(src="a", dst="c", type=EdgeType.RELATED))
        self.repo.upsert(Edge(src="b", dst="c", type=EdgeType.PREREQUISITE))
        all_out = self.repo.out_edges("a")
        self.assertEqual({e.dst for e in all_out}, {"b", "c"})
        only_prereq = self.repo.out_edges("a", EdgeType.PREREQUISITE)
        self.assertEqual([(e.src, e.dst) for e in only_prereq], [("a", "b")])

    def test_in_edges_filter_by_type(self) -> None:
        self.repo.upsert(Edge(src="a", dst="c", type=EdgeType.RELATED))
        self.repo.upsert(Edge(src="b", dst="c", type=EdgeType.PREREQUISITE))
        all_in = self.repo.in_edges("c")
        self.assertEqual({e.src for e in all_in}, {"a", "b"})
        only_related = self.repo.in_edges("c", EdgeType.RELATED)
        self.assertEqual([(e.src, e.dst) for e in only_related], [("a", "c")])

    def test_list_by_course_resolves_via_source_concept(self) -> None:
        self.concepts.upsert(_concept("x", course="other"))
        self.repo.upsert(Edge(src="a", dst="b", type=EdgeType.RELATED))      # cs101
        self.repo.upsert(Edge(src="x", dst="b", type=EdgeType.RELATED))      # other
        listed = self.repo.list_by_course("cs101")
        self.assertEqual([e.src for e in listed], ["a"])

    def test_record_exposure_traversal_sets_last_traversed(self) -> None:
        self.repo.upsert(Edge(src="a", dst="b", type=EdgeType.RELATED))
        self.repo.record_exposure("a", "b", EdgeType.RELATED, skipped=False, at=T0)
        e = self.repo.get("a", "b", EdgeType.RELATED)
        self.assertEqual(e.exposure_count, 1)
        self.assertEqual(e.skip_count, 0)
        self.assertEqual(e.last_traversed, T0)

    def test_record_exposure_skip_increments_skip_only(self) -> None:
        self.repo.upsert(Edge(src="a", dst="b", type=EdgeType.RELATED))
        self.repo.record_exposure("a", "b", EdgeType.RELATED, skipped=True, at=T0)
        e = self.repo.get("a", "b", EdgeType.RELATED)
        self.assertEqual(e.exposure_count, 1)
        self.assertEqual(e.skip_count, 1)
        # A skip is not a traversal: last_traversed must stay unset.
        self.assertIsNone(e.last_traversed)

    def test_record_exposure_accumulates(self) -> None:
        self.repo.upsert(Edge(src="a", dst="b", type=EdgeType.RELATED))
        self.repo.record_exposure("a", "b", EdgeType.RELATED, skipped=True, at=T0)
        self.repo.record_exposure("a", "b", EdgeType.RELATED, skipped=False, at=T0 + timedelta(days=1))
        self.repo.record_exposure("a", "b", EdgeType.RELATED, skipped=True, at=T0 + timedelta(days=2))
        e = self.repo.get("a", "b", EdgeType.RELATED)
        self.assertEqual(e.exposure_count, 3)
        self.assertEqual(e.skip_count, 2)
        # last_traversed reflects the most recent non-skipped exposure only.
        self.assertEqual(e.last_traversed, T0 + timedelta(days=1))

    def test_record_exposure_creates_edge_when_absent(self) -> None:
        # Total accounting: never fails just because ingestion hasn't run yet.
        self.repo.record_exposure("a", "b", EdgeType.RELATED, skipped=False, at=T0)
        e = self.repo.get("a", "b", EdgeType.RELATED)
        self.assertIsNotNone(e)
        self.assertEqual(e.exposure_count, 1)
        self.assertEqual(e.last_traversed, T0)

    def test_record_exposure_does_not_mutate_original(self) -> None:
        original = Edge(src="a", dst="b", type=EdgeType.RELATED)
        self.repo.upsert(original)
        self.repo.record_exposure("a", "b", EdgeType.RELATED, skipped=False, at=T0)
        # Functional replace: the originally-stored value object is untouched.
        self.assertEqual(original.exposure_count, 0)
        self.assertIsNone(original.last_traversed)


class QuestionRepositoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.repo = InMemoryQuestionRepository()

    def test_upsert_get_round_trip(self) -> None:
        q = Question(id="q1", concept_id="a")
        self.repo.upsert(q)
        self.assertEqual(self.repo.get("q1"), q)

    def test_get_missing_returns_none(self) -> None:
        self.assertIsNone(self.repo.get("q404"))

    def test_by_concept_filters(self) -> None:
        self.repo.upsert(Question(id="q1", concept_id="a", difficulty=1, hop_count=1))
        self.repo.upsert(Question(id="q2", concept_id="a", difficulty=3, hop_count=2))
        self.repo.upsert(Question(id="q3", concept_id="b", difficulty=1))
        self.assertEqual({q.id for q in self.repo.by_concept("a")}, {"q1", "q2"})
        self.assertEqual(
            [q.id for q in self.repo.by_concept("a", difficulty=3)], ["q2"]
        )
        self.assertEqual(
            [q.id for q in self.repo.by_concept("a", hop_count=1)], ["q1"]
        )
        self.assertEqual(
            self.repo.by_concept("a", difficulty=3, hop_count=1), []
        )

    def test_by_edge(self) -> None:
        self.repo.upsert(Question(id="q1", concept_id="a", edge_id="a::related::b"))
        self.repo.upsert(Question(id="q2", concept_id="a", edge_id="a::related::b"))
        self.repo.upsert(Question(id="q3", concept_id="a", edge_id=None))
        out = self.repo.by_edge("a::related::b")
        self.assertEqual([q.id for q in out], ["q1", "q2"])

    def test_by_edge_none_does_not_match_unlinked(self) -> None:
        self.repo.upsert(Question(id="q3", concept_id="a", edge_id=None))
        self.assertEqual(self.repo.by_edge("a::related::b"), [])


class LearnerStateRepositoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.concepts = InMemoryConceptIndexRepository()
        self.concepts.upsert(_concept("a", course="cs101"))
        self.concepts.upsert(_concept("b", course="cs101"))
        self.concepts.upsert(_concept("x", course="other"))
        self.repo = InMemoryLearnerStateRepository(self.concepts)

    def test_upsert_get_round_trip(self) -> None:
        s = LearnerState(concept_id="a", stability=10.0, due_at=T0)
        self.repo.upsert(s)
        self.assertEqual(self.repo.get("a"), s)

    def test_get_missing_returns_none(self) -> None:
        self.assertIsNone(self.repo.get("a"))

    def test_due_returns_only_past_due_in_course(self) -> None:
        self.repo.upsert(LearnerState(concept_id="a", due_at=T0 - timedelta(days=1)))
        self.repo.upsert(LearnerState(concept_id="b", due_at=T0 + timedelta(days=1)))
        self.repo.upsert(LearnerState(concept_id="x", due_at=T0 - timedelta(days=5)))
        due = self.repo.due("cs101", before=T0)
        # Only 'a': 'b' is in the future, 'x' belongs to another course.
        self.assertEqual([s.concept_id for s in due], ["a"])

    def test_due_boundary_is_inclusive(self) -> None:
        self.repo.upsert(LearnerState(concept_id="a", due_at=T0))
        due = self.repo.due("cs101", before=T0)
        self.assertEqual([s.concept_id for s in due], ["a"])

    def test_due_excludes_unscheduled_states(self) -> None:
        self.repo.upsert(LearnerState(concept_id="a", due_at=None))
        self.assertEqual(self.repo.due("cs101", before=T0), [])

    def test_due_sorted_by_due_at(self) -> None:
        self.repo.upsert(LearnerState(concept_id="b", due_at=T0 - timedelta(days=1)))
        self.repo.upsert(LearnerState(concept_id="a", due_at=T0 - timedelta(days=2)))
        due = self.repo.due("cs101", before=T0)
        self.assertEqual([s.concept_id for s in due], ["a", "b"])

    def test_all_for_course_resolves_via_concept(self) -> None:
        self.repo.upsert(LearnerState(concept_id="a"))
        self.repo.upsert(LearnerState(concept_id="b"))
        self.repo.upsert(LearnerState(concept_id="x"))
        out = self.repo.all_for_course("cs101")
        self.assertEqual([s.concept_id for s in out], ["a", "b"])

    def test_states_with_unknown_concept_are_excluded(self) -> None:
        self.repo.upsert(LearnerState(concept_id="ghost", due_at=T0 - timedelta(days=1)))
        self.assertEqual(self.repo.all_for_course("cs101"), [])
        self.assertEqual(self.repo.due("cs101", before=T0), [])


class ReviewLogRepositoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.repo = InMemoryReviewLogRepository()

    def _event(self, cid: str, grade: int, at: datetime) -> ReviewEvent:
        return ReviewEvent(
            concept_id=cid, grade=grade, fsrs_rating=FsrsRating.GOOD, at=at
        )

    def test_append_and_by_concept_round_trip(self) -> None:
        e = self._event("a", 5, T0)
        self.repo.append(e)
        self.assertEqual(self.repo.by_concept("a"), [e])

    def test_by_concept_missing_is_empty(self) -> None:
        self.assertEqual(self.repo.by_concept("a"), [])

    def test_by_concept_preserves_append_order_and_filters(self) -> None:
        e1 = self._event("a", 3, T0)
        e2 = self._event("b", 4, T0 + timedelta(hours=1))
        e3 = self._event("a", 6, T0 + timedelta(hours=2))
        for e in (e1, e2, e3):
            self.repo.append(e)
        self.assertEqual(self.repo.by_concept("a"), [e1, e3])


class CourseProfileRepositoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.repo = InMemoryCourseProfileRepository()

    def test_upsert_get_round_trip(self) -> None:
        p = CourseProfile(course="cs101", archetype="conceptual-written")
        self.repo.upsert(p)
        self.assertEqual(self.repo.get("cs101"), p)

    def test_get_missing_returns_none(self) -> None:
        self.assertIsNone(self.repo.get("nope"))

    def test_upsert_overwrites(self) -> None:
        self.repo.upsert(CourseProfile(course="cs101", archetype="a"))
        self.repo.upsert(CourseProfile(course="cs101", archetype="b"))
        self.assertEqual(self.repo.get("cs101").archetype, "b")


class ContentRepositoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.repo = InMemoryContentRepository()

    def test_concept_content_round_trip(self) -> None:
        c = ConceptContent(concept_id="a", title="A", body="hello")
        self.repo.put_concept_content(c)
        self.assertEqual(self.repo.get_concept_content("a"), c)

    def test_get_missing_concept_content_returns_none(self) -> None:
        self.assertIsNone(self.repo.get_concept_content("a"))

    def test_question_content_round_trip(self) -> None:
        q = QuestionContent(question_id="q1", prompt="why?", rubric="because")
        self.repo.put_question_content(q)
        self.assertEqual(self.repo.get_question_content("q1"), q)

    def test_get_missing_question_content_returns_none(self) -> None:
        self.assertIsNone(self.repo.get_question_content("q1"))

    def test_concept_hash_is_hex_sha256(self) -> None:
        digest = self.repo.put_concept_content(
            ConceptContent(concept_id="a", title="A", body="b")
        )
        self.assertEqual(len(digest), 64)
        int(digest, 16)  # raises if not valid hex

    def test_concept_hash_is_stable(self) -> None:
        c = ConceptContent(
            concept_id="a",
            title="A",
            body="b",
            source_refs=(SourceRef(file="f.md", line=3),),
        )
        first = self.repo.put_concept_content(c)
        # Re-serialising identical content yields the identical digest.
        again = self.repo.put_concept_content(
            ConceptContent(
                concept_id="a",
                title="A",
                body="b",
                source_refs=(SourceRef(file="f.md", line=3),),
            )
        )
        self.assertEqual(first, again)

    def test_concept_hash_changes_with_content(self) -> None:
        h1 = self.repo.put_concept_content(ConceptContent(concept_id="a", title="A", body="one"))
        h2 = self.repo.put_concept_content(ConceptContent(concept_id="a", title="A", body="two"))
        self.assertNotEqual(h1, h2)

    def test_concept_hash_changes_with_source_refs(self) -> None:
        h1 = self.repo.put_concept_content(
            ConceptContent(concept_id="a", title="A", body="b", source_refs=())
        )
        h2 = self.repo.put_concept_content(
            ConceptContent(
                concept_id="a",
                title="A",
                body="b",
                source_refs=(SourceRef(file="f.md", line=1),),
            )
        )
        self.assertNotEqual(h1, h2)

    def test_question_hash_stable_and_content_sensitive(self) -> None:
        h1 = self.repo.put_question_content(QuestionContent(question_id="q1", prompt="p"))
        h1_again = self.repo.put_question_content(QuestionContent(question_id="q1", prompt="p"))
        h2 = self.repo.put_question_content(QuestionContent(question_id="q1", prompt="p2"))
        self.assertEqual(h1, h1_again)
        self.assertNotEqual(h1, h2)

    def test_put_returns_same_hash_get_would_imply(self) -> None:
        c = ConceptContent(concept_id="a", title="A", body="b")
        returned = self.repo.put_concept_content(c)
        listed = dict(self.repo.iter_concepts())
        self.assertEqual(listed["a"], returned)

    def test_iter_concepts_yields_id_hash_pairs_sorted(self) -> None:
        hb = self.repo.put_concept_content(ConceptContent(concept_id="b", title="B", body="bb"))
        ha = self.repo.put_concept_content(ConceptContent(concept_id="a", title="A", body="aa"))
        pairs = list(self.repo.iter_concepts())
        self.assertEqual([cid for cid, _ in pairs], ["a", "b"])
        self.assertEqual(dict(pairs), {"a": ha, "b": hb})

    def test_iter_concepts_reflects_latest_hash_after_update(self) -> None:
        self.repo.put_concept_content(ConceptContent(concept_id="a", title="A", body="old"))
        new = self.repo.put_concept_content(ConceptContent(concept_id="a", title="A", body="new"))
        self.assertEqual(dict(self.repo.iter_concepts())["a"], new)


if __name__ == "__main__":
    unittest.main()

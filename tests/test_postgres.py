"""Tests for the Postgres + pgvector repository adapters.

Two layers:

1. Driver-free tests that ALWAYS run: the module must import even with no
   psycopg/pgvector installed, the pure mapping helpers must round-trip, the
   concrete adapters must subclass the right ports, and ``connect`` must fail
   loudly when the driver is absent. These pin the import-without-driver
   contract and the JSON/vector mapping invariants without touching a database.

2. Live round-trip tests gated behind ``CURRICULUM_PG_TEST`` + a real
   ``CURRICULUM_DB_URL``. They apply schema/001_init.sql and exercise concept,
   edge, and learner_state persistence (plus the JOIN-based course scoping and
   the pgvector nearest-neighbour search) against a real Postgres.
"""
from __future__ import annotations

import os
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from curriculum.domain.entities import (
    Concept,
    CourseProfile,
    Edge,
    LearnerState,
    Question,
    ReviewEvent,
    SourceRef,
)
from curriculum.domain.enums import EdgeType, FsrsRating, Mastery
from curriculum.ports.repositories import (
    ConceptIndexRepository,
    CourseProfileRepository,
    EdgeRepository,
    LearnerStateRepository,
    QuestionRepository,
    ReviewLogRepository,
)
from curriculum.storage import postgres
from curriculum.storage.postgres import (
    PostgresConceptIndexRepository,
    PostgresCourseProfileRepository,
    PostgresEdgeRepository,
    PostgresLearnerStateRepository,
    PostgresQuestionRepository,
    PostgresRepositories,
    PostgresReviewLogRepository,
    connect,
    psycopg,
)

_SCHEMA = Path(__file__).resolve().parents[1] / "schema" / "001_init.sql"
_LIVE = psycopg is not None and bool(os.environ.get("CURRICULUM_PG_TEST"))


# --------------------------------------------------------------------------- #
# Always-on: import contract, mapping helpers, wiring, driver guard.
# --------------------------------------------------------------------------- #
class ImportContractTest(unittest.TestCase):
    def test_module_imported_without_driver(self) -> None:
        # The import at the top of this file already proves the module loads;
        # assert the public surface is present regardless of the driver.
        self.assertTrue(hasattr(postgres, "PostgresRepositories"))
        self.assertTrue(hasattr(postgres, "connect"))

    def test_concrete_adapters_subclass_ports(self) -> None:
        self.assertTrue(issubclass(PostgresConceptIndexRepository, ConceptIndexRepository))
        self.assertTrue(issubclass(PostgresEdgeRepository, EdgeRepository))
        self.assertTrue(issubclass(PostgresQuestionRepository, QuestionRepository))
        self.assertTrue(issubclass(PostgresLearnerStateRepository, LearnerStateRepository))
        self.assertTrue(issubclass(PostgresReviewLogRepository, ReviewLogRepository))
        self.assertTrue(issubclass(PostgresCourseProfileRepository, CourseProfileRepository))


class MappingHelperTest(unittest.TestCase):
    def test_vector_literal_pgvector_text_format(self) -> None:
        self.assertEqual(postgres._vector_literal([1.0, 0.5, 2.0]), "[1.0,0.5,2.0]")

    def test_vector_literal_coerces_ints_to_float(self) -> None:
        self.assertEqual(postgres._vector_literal([1, 2, 3]), "[1.0,2.0,3.0]")

    def test_vector_literal_accepts_any_sequence(self) -> None:
        self.assertEqual(postgres._vector_literal((0.0, -1.5)), "[0.0,-1.5]")

    def test_refs_round_trip(self) -> None:
        refs = (SourceRef(file="f.md", line=3), SourceRef(file="g.md", line=None))
        self.assertEqual(postgres._refs_from_json(postgres._refs_to_json(refs)), refs)

    def test_refs_from_json_handles_null_and_empty(self) -> None:
        self.assertEqual(postgres._refs_from_json(None), ())
        self.assertEqual(postgres._refs_from_json([]), ())

    def test_single_ref_round_trip(self) -> None:
        ref = SourceRef(file="f.md", line=7)
        self.assertEqual(postgres._ref_from_json(postgres._ref_to_json(ref)), ref)

    def test_single_ref_none_round_trip(self) -> None:
        self.assertIsNone(postgres._ref_to_json(None))
        self.assertIsNone(postgres._ref_from_json(None))

    def test_similarity_from_distance_is_one_at_zero(self) -> None:
        self.assertEqual(postgres._similarity_from_distance(0.0), 1.0)

    def test_similarity_from_distance_monotone_decreasing_in_unit_range(self) -> None:
        s0 = postgres._similarity_from_distance(0.0)
        s1 = postgres._similarity_from_distance(1.0)
        s5 = postgres._similarity_from_distance(5.0)
        self.assertGreater(s0, s1)
        self.assertGreater(s1, s5)
        # higher == closer, bounded to (0, 1]
        for s in (s0, s1, s5):
            self.assertGreater(s, 0.0)
            self.assertLessEqual(s, 1.0)


@unittest.skipUnless(
    postgres.register_vector is None,
    "container smoke test must not register pgvector on a non-connection",
)
class ContainerWiringTest(unittest.TestCase):
    """When the driver is absent, the container can be built with a stand-in
    connection: no SQL runs at construction, and vector registration is skipped.
    This verifies the wiring (six repositories, correct types) cheaply."""

    def setUp(self) -> None:
        self.repos = PostgresRepositories(conn=object())

    def test_exposes_six_named_repositories(self) -> None:
        self.assertIsInstance(self.repos.concepts, ConceptIndexRepository)
        self.assertIsInstance(self.repos.edges, EdgeRepository)
        self.assertIsInstance(self.repos.questions, QuestionRepository)
        self.assertIsInstance(self.repos.learner_state, LearnerStateRepository)
        self.assertIsInstance(self.repos.review_log, ReviewLogRepository)
        self.assertIsInstance(self.repos.profiles, CourseProfileRepository)

    def test_repositories_share_the_one_connection(self) -> None:
        conn = object()
        repos = PostgresRepositories(conn=conn)
        self.assertIs(repos.concepts._conn, conn)
        self.assertIs(repos.edges._conn, conn)
        self.assertIs(repos.learner_state._conn, conn)


@unittest.skipUnless(psycopg is None, "guard message only applies without the driver")
class ConnectGuardTest(unittest.TestCase):
    def test_connect_raises_clear_error_without_driver(self) -> None:
        with self.assertRaises(RuntimeError):
            connect("postgresql://localhost/whatever")


# --------------------------------------------------------------------------- #
# Live round-trip tests (need psycopg + a real Postgres with pgvector).
# --------------------------------------------------------------------------- #
@unittest.skipUnless(_LIVE, "needs psycopg + Postgres (set CURRICULUM_PG_TEST and CURRICULUM_DB_URL)")
class PostgresLiveTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.conn = connect(os.environ["CURRICULUM_DB_URL"])
        # DDL script: no params -> psycopg can run the multi-statement file.
        cls.conn.execute(_SCHEMA.read_text())
        cls.conn.commit()
        cls.repos = PostgresRepositories(cls.conn)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.conn.close()

    def setUp(self) -> None:
        # A unique course per test isolates rows; concept-delete cascades clean
        # up edges/questions/state/log via the schema's ON DELETE CASCADE.
        self.course = "pgtest_" + uuid.uuid4().hex[:12]

    def tearDown(self) -> None:
        self.conn.execute("DELETE FROM concept WHERE course = %s", (self.course,))
        self.conn.commit()

    def _concept(self, suffix: str, course: str | None = None, **kw) -> Concept:
        return Concept(
            id=f"{self.course}:{suffix}",
            course=course or self.course,
            title=suffix.title(),
            **kw,
        )

    def test_concept_round_trip(self) -> None:
        c = self._concept(
            "a",
            description="d",
            importance=0.75,
            source_refs=(SourceRef(file="f.md", line=3),),
            content_hash="hash123",
            status="active",
        )
        self.repos.concepts.upsert(c)
        self.conn.commit()
        self.assertEqual(self.repos.concepts.get(c.id), c)

    def test_concept_get_missing_is_none(self) -> None:
        self.assertIsNone(self.repos.concepts.get(f"{self.course}:ghost"))

    def test_concept_list_by_course_sorted(self) -> None:
        self.repos.concepts.upsert(self._concept("b"))
        self.repos.concepts.upsert(self._concept("a"))
        self.conn.commit()
        listed = self.repos.concepts.list_by_course(self.course)
        self.assertEqual([c.id for c in listed], [f"{self.course}:a", f"{self.course}:b"])

    def test_upsert_preserves_embedding(self) -> None:
        c = self._concept("a")
        self.repos.concepts.upsert(c)
        self.repos.concepts.set_embedding(c.id, [1.0] + [0.0] * 1023)
        # A metadata re-upsert must not wipe the derived embedding.
        self.repos.concepts.upsert(self.repos.concepts.get(c.id))
        self.conn.commit()
        nearest = self.repos.concepts.nearest(
            [1.0] + [0.0] * 1023, course=self.course, k=1
        )
        self.assertEqual(nearest[0][0], c.id)

    def test_nearest_orders_by_distance_and_returns_similarity(self) -> None:
        near = self._concept("near")
        far = self._concept("far")
        self.repos.concepts.upsert(near)
        self.repos.concepts.upsert(far)
        query = [1.0] + [0.0] * 1023
        self.repos.concepts.set_embedding(near.id, query)            # distance 0
        self.repos.concepts.set_embedding(far.id, [0.0, 1.0] + [0.0] * 1022)  # distance sqrt(2)
        self.conn.commit()
        result = self.repos.concepts.nearest(query, course=self.course, k=2)
        self.assertEqual([cid for cid, _ in result], [near.id, far.id])
        self.assertAlmostEqual(result[0][1], 1.0)          # exact hit -> similarity 1.0
        self.assertGreater(result[0][1], result[1][1])     # higher == closer

    def test_nearest_non_positive_k_is_empty(self) -> None:
        c = self._concept("a")
        self.repos.concepts.upsert(c)
        self.repos.concepts.set_embedding(c.id, [1.0] + [0.0] * 1023)
        self.conn.commit()
        self.assertEqual(
            self.repos.concepts.nearest([1.0] + [0.0] * 1023, course=self.course, k=0),
            [],
        )

    def test_edge_round_trip(self) -> None:
        a, b = self._concept("a"), self._concept("b")
        self.repos.concepts.upsert(a)
        self.repos.concepts.upsert(b)
        e = Edge(
            src=a.id,
            dst=b.id,
            type=EdgeType.PREREQUISITE,
            weight=0.5,
            importance=0.25,
            rationale="needs a first",
            source_ref=SourceRef(file="f.md", line=1),
        )
        self.repos.edges.upsert(e)
        self.conn.commit()
        self.assertEqual(self.repos.edges.get(a.id, b.id, EdgeType.PREREQUISITE), e)
        self.assertEqual([x.id for x in self.repos.edges.out_edges(a.id)], [e.id])
        self.assertEqual([x.id for x in self.repos.edges.in_edges(b.id)], [e.id])
        self.assertEqual([x.id for x in self.repos.edges.list_by_course(self.course)], [e.id])

    def test_edge_get_discriminates_by_type(self) -> None:
        a, b = self._concept("a"), self._concept("b")
        self.repos.concepts.upsert(a)
        self.repos.concepts.upsert(b)
        self.repos.edges.upsert(Edge(src=a.id, dst=b.id, type=EdgeType.PREREQUISITE))
        self.conn.commit()
        self.assertIsNone(self.repos.edges.get(a.id, b.id, EdgeType.RELATED))

    def test_record_exposure_accumulates_and_tracks_traversal(self) -> None:
        a, b = self._concept("a"), self._concept("b")
        self.repos.concepts.upsert(a)
        self.repos.concepts.upsert(b)
        t0 = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
        # Edge does not exist yet -> the upsert keeps the operation total.
        self.repos.edges.record_exposure(a.id, b.id, EdgeType.RELATED, skipped=False, at=t0)
        self.repos.edges.record_exposure(
            a.id, b.id, EdgeType.RELATED, skipped=True, at=t0 + timedelta(days=1)
        )
        self.conn.commit()
        e = self.repos.edges.get(a.id, b.id, EdgeType.RELATED)
        self.assertEqual(e.exposure_count, 2)
        self.assertEqual(e.skip_count, 1)
        # last_traversed reflects the most recent NON-skipped exposure (t0).
        self.assertEqual(e.last_traversed, t0)

    def test_learner_state_round_trip_and_course_scoping(self) -> None:
        a = self._concept("a")
        other = self._concept("x", course=self.course + "_other")
        self.repos.concepts.upsert(a)
        self.repos.concepts.upsert(other)
        t0 = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
        s = LearnerState(
            concept_id=a.id,
            stability=10.0,
            difficulty=5.0,
            last_review=t0,
            due_at=t0,
            reps=2,
            lapses=1,
            mastery=Mastery.LEARNING,
        )
        # A due state in another course must be excluded by the JOIN filter.
        self.repos.learner_state.upsert(s)
        self.repos.learner_state.upsert(
            LearnerState(concept_id=other.id, due_at=t0 - timedelta(days=1))
        )
        self.conn.commit()
        self.assertEqual(self.repos.learner_state.get(a.id), s)
        due = self.repos.learner_state.due(self.course, before=t0)
        self.assertEqual([x.concept_id for x in due], [a.id])
        allc = self.repos.learner_state.all_for_course(self.course)
        self.assertEqual([x.concept_id for x in allc], [a.id])
        # cleanup for the extra course used here
        self.conn.execute("DELETE FROM concept WHERE course = %s", (self.course + "_other",))
        self.conn.commit()

    def test_question_round_trip_and_filters(self) -> None:
        a = self._concept("a")
        self.repos.concepts.upsert(a)
        q1 = Question(id=f"{self.course}:q1", concept_id=a.id, difficulty=1, hop_count=1)
        q2 = Question(
            id=f"{self.course}:q2",
            concept_id=a.id,
            difficulty=3,
            hop_count=2,
            edge_id="e::related::f",
            source_refs=(SourceRef(file="f.md", line=2),),
        )
        self.repos.questions.upsert(q1)
        self.repos.questions.upsert(q2)
        self.conn.commit()
        self.assertEqual(self.repos.questions.get(q2.id), q2)
        self.assertEqual(
            {q.id for q in self.repos.questions.by_concept(a.id)}, {q1.id, q2.id}
        )
        self.assertEqual(
            [q.id for q in self.repos.questions.by_concept(a.id, difficulty=3)], [q2.id]
        )
        self.assertEqual(
            [q.id for q in self.repos.questions.by_concept(a.id, hop_count=1)], [q1.id]
        )
        self.assertEqual([q.id for q in self.repos.questions.by_edge("e::related::f")], [q2.id])

    def test_review_log_append_preserves_order(self) -> None:
        a = self._concept("a")
        self.repos.concepts.upsert(a)
        t0 = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
        e1 = ReviewEvent(concept_id=a.id, grade=3, fsrs_rating=FsrsRating.GOOD, at=t0)
        e2 = ReviewEvent(
            concept_id=a.id,
            grade=6,
            fsrs_rating=FsrsRating.EASY,
            at=t0 + timedelta(hours=1),
            question_id=f"{self.course}:q1",
            predicted=5,
        )
        self.repos.review_log.append(e1)
        self.repos.review_log.append(e2)
        self.conn.commit()
        got = self.repos.review_log.by_concept(a.id)
        self.assertEqual([ev.grade for ev in got], [3, 6])
        self.assertEqual(got[1], e2)

    def test_course_profile_round_trip(self) -> None:
        a = self._concept("a")  # not strictly needed, keeps tearDown symmetric
        self.repos.concepts.upsert(a)
        p = CourseProfile(
            course=self.course,
            archetype="conceptual-written",
            exam_format={"sections": 3},
            weights={"urgency": 1.0, "coverage": 0.5},
            target_retention=0.9,
            confirmed_by_user=True,
        )
        self.repos.profiles.upsert(p)
        self.conn.commit()
        got = self.repos.profiles.get(self.course)
        self.assertEqual(got.archetype, "conceptual-written")
        self.assertEqual(dict(got.exam_format), {"sections": 3})
        self.assertEqual(dict(got.weights), {"urgency": 1.0, "coverage": 0.5})
        self.assertTrue(got.confirmed_by_user)
        # profile has no FK to concept; clean it explicitly.
        self.conn.execute("DELETE FROM course_profile WHERE course = %s", (self.course,))
        self.conn.commit()


if __name__ == "__main__":
    unittest.main()

"""Motivation-layer use-cases on the application service.

Drives checkin (memory-capital totals + delta since the last check), frontier
(the three strategy buckets), flag_question (the kill switch), and the grade
ripple report through the public CurriculumService on the in-memory stack.
A FixedClock keeps every number deterministic; telemetry is asserted through
the stack's InMemoryTelemetryRepository handle.
"""
from __future__ import annotations

import unittest
from datetime import date, datetime, timedelta

from curriculum.application.composition import build_in_memory
from curriculum.application.policies import FixedClock
from curriculum.domain.entities import (
    Concept,
    ConceptContent,
    CourseProfile,
    Edge,
    LearnerState,
    Question,
    QuestionContent,
    SourceRef,
)
from curriculum.domain.enums import EdgeType, Mastery
from curriculum.domain.errors import QuestionNotFound

COURSE = "Cybersecurity"
NOW = datetime(2026, 7, 1, 9, 0, 0)


class MotivationServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.stack = build_in_memory(clock=FixedClock(NOW))
        s = self.stack

        s.profiles.upsert(
            CourseProfile(
                course=COURSE,
                archetype="conceptual-written",
                exam_date=date(2026, 7, 21),
                target_retention=0.9,
            )
        )

        # cia (root) -- prerequisite of aes; encompasses confidentiality (FIRe).
        # hashing is a second root with higher importance (the best TEACH pick).
        for cid, title, importance in [
            ("cyber/cia", "CIA Triad", 0.5),
            ("cyber/aes", "AES", 0.5),
            ("cyber/hashing", "Hashing", 0.8),
            ("cyber/confidentiality", "Confidentiality", 0.5),
        ]:
            s.concepts.upsert(
                Concept(
                    id=cid, course=COURSE, title=title, importance=importance,
                    source_refs=(SourceRef("lessons/cyber.md", 1),),
                )
            )
            s.content.put_concept_content(
                ConceptContent(concept_id=cid, title=title, body=f"Body of {title}.",
                               source_refs=(SourceRef("lessons/cyber.md", 1),))
            )

        s.edges.upsert(Edge(src="cyber/cia", dst="cyber/aes", type=EdgeType.PREREQUISITE))
        s.edges.upsert(
            Edge(src="cyber/cia", dst="cyber/confidentiality",
                 type=EdgeType.ENCOMPASSES, weight=0.9)
        )

        s.questions.upsert(Question(id="q-cia", concept_id="cyber/cia"))
        s.content.put_question_content(
            QuestionContent("q-cia", "Define the CIA triad.", "names C/I/A")
        )

    # ------------------------------------------------------------- helpers
    def _events(self, kind: str) -> list:
        return [e for e in self.stack.telemetry.list_by_course(COURSE) if e.kind == kind]

    def _due_learning_state_on_cia(self) -> None:
        """A started-but-unmastered cia with a review ready now."""
        self.stack.states.upsert(
            LearnerState(
                concept_id="cyber/cia",
                stability=1.0,
                difficulty=5.0,
                last_review=NOW - timedelta(days=3),
                due_at=NOW - timedelta(days=1),
                reps=1,
                mastery=Mastery.LEARNING,
            )
        )

    # ------------------------------------------------------------- checkin
    def test_first_checkin_has_no_delta_and_logs_one_check_event(self) -> None:
        out = self.stack.service.checkin(COURSE)

        self.assertEqual(out["course"], COURSE)
        self.assertEqual(out["stability_days"], 0.0)
        self.assertIsNone(out["delta_since_last_check"])
        for key in ("consolidation", "ripeness", "unlocks_ready", "near_unlocks", "by_mastery"):
            self.assertIn(key, out)

        checks = self._events("check")
        self.assertEqual(len(checks), 1)
        self.assertEqual(checks[0].payload["stability_days"], 0.0)

    def test_second_checkin_reports_positive_delta_after_a_graded_review(self) -> None:
        first = self.stack.service.checkin(COURSE)
        self.assertEqual(first["stability_days"], 0.0)

        self.stack.service.grade(concept_id="cyber/cia", score=6)

        second = self.stack.service.checkin(COURSE)
        self.assertIsNotNone(second["delta_since_last_check"])
        self.assertGreater(second["delta_since_last_check"], 0.0)
        self.assertAlmostEqual(
            second["delta_since_last_check"], second["stability_days"] - 0.0
        )
        self.assertEqual(len(self._events("check")), 2)

    def test_checkin_reports_unlocks_ready_and_near_unlocks(self) -> None:
        out = self.stack.service.checkin(COURSE)
        # all roots are ready to start; aes is gated by an untouched cia
        self.assertEqual(
            out["unlocks_ready"],
            ["cyber/cia", "cyber/confidentiality", "cyber/hashing"],
        )
        self.assertEqual(len(out["near_unlocks"]), 1)
        row = out["near_unlocks"][0]
        self.assertEqual(row["concept_id"], "cyber/aes")
        self.assertEqual(row["missing"], 1)
        self.assertFalse(row["one_away"])

        # mastering cia unlocks aes (and FIRe marks confidentiality as started)
        self.stack.service.grade(concept_id="cyber/cia", score=6)
        after = self.stack.service.checkin(COURSE)
        self.assertIn("cyber/aes", after["unlocks_ready"])
        self.assertEqual(after["near_unlocks"], [])
        self.assertEqual(after["by_mastery"]["solid"], 1)

    # ------------------------------------------------------------- frontier
    def test_frontier_returns_three_distinct_buckets(self) -> None:
        self._due_learning_state_on_cia()

        out = self.stack.service.frontier(COURSE)

        self.assertEqual(set(out), {"push", "reinforce", "breakthrough"})
        for entry in out.values():
            self.assertEqual(set(entry), {"concept_id", "mode", "reason", "score"})
            self.assertIsInstance(entry["score"], float)

        self.assertEqual(out["push"]["concept_id"], "cyber/hashing")
        self.assertEqual(out["push"]["mode"], "teach")
        self.assertEqual(out["reinforce"]["concept_id"], "cyber/cia")
        self.assertEqual(out["reinforce"]["mode"], "review")
        self.assertEqual(out["breakthrough"]["concept_id"], "cyber/aes")

        ids = {entry["concept_id"] for entry in out.values()}
        self.assertEqual(len(ids), 3)

        escalations = self._events("escalate")
        self.assertEqual(len(escalations), 1)

    def test_frontier_does_not_mutate_interleaving_memory(self) -> None:
        self._due_learning_state_on_cia()
        self.stack.service.frontier(COURSE)
        self.assertEqual(self.stack.service._last_cluster, {})

    def test_frontier_omits_empty_buckets(self) -> None:
        # focus scopes to hashing only: one TEACH candidate, nothing to review,
        # no in-scope locked concept -> only "push" survives
        out = self.stack.service.frontier(COURSE, focus="hashing")
        self.assertEqual(set(out), {"push"})
        self.assertEqual(out["push"]["concept_id"], "cyber/hashing")

    def test_frontier_with_no_matches_returns_no_buckets(self) -> None:
        out = self.stack.service.frontier(COURSE, focus="no-such-topic")
        self.assertEqual(dict(out), {})

    # -------------------------------------------------------- flag_question
    def test_flag_question_retires_it_and_logs_item_flag(self) -> None:
        # sanity: the question is served before the flag
        q, _qc = self.stack.service.quiz("cyber/cia")
        self.assertEqual(q.id, "q-cia")

        out = self.stack.service.flag_question("q-cia", reason="ambiguous stem")
        self.assertEqual(out, {"question_id": "q-cia", "status": "retired"})

        with self.assertRaises(QuestionNotFound):
            self.stack.service.quiz("cyber/cia")

        flags = self._events("item_flag")
        self.assertEqual(len(flags), 1)
        self.assertEqual(flags[0].payload["question_id"], "q-cia")
        self.assertEqual(flags[0].payload["reason"], "ambiguous stem")

    def test_flag_unknown_question_raises(self) -> None:
        with self.assertRaises(QuestionNotFound):
            self.stack.service.flag_question("q-nope")

    # ------------------------------------------------------------- ripple
    def test_grade_ripple_reports_fire_credited_stability_gain(self) -> None:
        out = self.stack.service.grade(concept_id="cyber/cia", score=6)

        # the 0.9-weight ENCOMPASSES edge earns confidentiality implicit credit
        self.assertEqual(len(out["fire_credits"]), 1)
        self.assertEqual(out["ripple"]["count"], 1)
        self.assertGreater(out["ripple"]["stability_days_gained"], 0.0)

        # importance-weighted exactly as snapshot.stability_days weights: both
        # concepts started from no retention signal (prior treated as 0.0)
        cia = self.stack.states.get("cyber/cia")
        conf = self.stack.states.get("cyber/confidentiality")
        expected = 0.5 * cia.stability + 0.5 * conf.stability
        self.assertAlmostEqual(out["ripple"]["stability_days_gained"], expected)

    def test_grade_ripple_without_fire_counts_only_the_primary_delta(self) -> None:
        out = self.stack.service.grade(concept_id="cyber/hashing", score=6)
        self.assertEqual(out["ripple"]["count"], 0)
        st = self.stack.states.get("cyber/hashing")
        self.assertAlmostEqual(out["ripple"]["stability_days_gained"], 0.8 * st.stability)


if __name__ == "__main__":
    unittest.main()

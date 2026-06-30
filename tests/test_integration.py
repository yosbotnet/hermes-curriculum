"""End-to-end integration test: the full engine on the in-memory stack.

Drives the real FSRS scheduler, scoring terms, weighted-sampling selection,
FIRe wiring, fringe gating, connection-skip escalation, mastery ladder, and
calibration logging through the public CurriculumService -- no mocks, only the
in-memory adapters. A FixedClock makes every assertion deterministic.
"""
from __future__ import annotations

import unittest
from datetime import date, datetime

from curriculum.application.composition import build_in_memory
from curriculum.application.policies import FixedClock
from curriculum.domain.entities import (
    Concept,
    ConceptContent,
    CourseProfile,
    Edge,
    Question,
    QuestionContent,
    SourceRef,
)
from curriculum.domain.enums import EdgeType, NextMode

COURSE = "Cybersecurity"
NOW = datetime(2026, 7, 1, 9, 0, 0)


class FullStackTests(unittest.TestCase):
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

        # three concepts: cia (root), aes (needs cia), confidentiality (related to cia)
        for cid, title in [
            ("cyber/cia", "CIA Triad"),
            ("cyber/aes", "AES"),
            ("cyber/confidentiality", "Confidentiality"),
        ]:
            s.concepts.upsert(
                Concept(id=cid, course=COURSE, title=title,
                        source_refs=(SourceRef("lessons/cyber.md", 1),))
            )
            s.content.put_concept_content(
                ConceptContent(concept_id=cid, title=title, body=f"Body of {title}.",
                               source_refs=(SourceRef("lessons/cyber.md", 1),))
            )

        # aes requires cia (prerequisite gate); cia relates-to confidentiality (skip target)
        self.prereq = Edge(src="cyber/cia", dst="cyber/aes", type=EdgeType.PREREQUISITE)
        self.related = Edge(src="cyber/cia", dst="cyber/confidentiality",
                            type=EdgeType.RELATED, importance=0.9)
        s.edges.upsert(self.prereq)
        s.edges.upsert(self.related)

        # a question on cia, and a multi-hop question on the related edge
        s.questions.upsert(Question(id="q-cia", concept_id="cyber/cia"))
        s.questions.upsert(
            Question(id="q-link", concept_id="cyber/cia", edge_id=self.related.id, hop_count=2)
        )
        s.content.put_question_content(QuestionContent("q-cia", "Define the CIA triad.", "names C/I/A"))
        s.content.put_question_content(
            QuestionContent("q-link", "How does the CIA triad relate to confidentiality?", "links them")
        )

    def test_next_starts_with_a_learnable_teach(self) -> None:
        result = self.stack.service.next_action(COURSE)
        # only cia and confidentiality are learnable; aes is gated behind cia
        ids = {c.concept_id for c in result.candidates}
        self.assertIn("cyber/cia", ids)
        self.assertNotIn("cyber/aes", ids)  # prerequisite not yet mastered
        self.assertEqual(result.chosen.mode, NextMode.TEACH)

    def test_explain_and_quiz_return_grounded_content(self) -> None:
        content = self.stack.service.explain("cyber/cia")
        self.assertIn("CIA", content.body)
        q, qc = self.stack.service.quiz("cyber/cia")
        self.assertEqual(q.id, "q-cia")
        self.assertIn("CIA", qc.prompt)

    def test_grade_updates_state_and_opens_the_gate(self) -> None:
        out = self.stack.service.grade(concept_id="cyber/cia", score=6, predicted=5)
        self.assertEqual(out["mastery"], "solid")
        self.assertIsNotNone(out["due_at"])

        st = self.stack.states.get("cyber/cia")
        self.assertIsNotNone(st)
        self.assertEqual(st.reps, 1)
        self.assertGreater(st.due_at, NOW)  # scheduled into the future

        # calibration: the predicted score was logged
        events = self.stack.reviews.by_concept("cyber/cia")
        self.assertEqual(events[-1].predicted, 5)

        # cia is now mastered -> aes becomes learnable in the next decision
        ids = {c.concept_id for c in self.stack.service.next_action(COURSE).candidates}
        self.assertIn("cyber/aes", ids)

    def test_repeated_connection_skip_escalates_to_a_test(self) -> None:
        # establish cia, then skip the cia<->confidentiality link three times
        self.stack.service.grade(concept_id="cyber/cia", score=6)
        for _ in range(3):
            out = self.stack.service.grade(
                concept_id="cyber/cia", score=5, skipped_edges=(self.related.id,)
            )
        self.assertIn(self.related.id, out["escalated_connections"])

        edge = self.stack.edges.get("cyber/cia", "cyber/confidentiality", EdgeType.RELATED)
        self.assertGreaterEqual(edge.skip_count, 3)

        # the engine should now surface a forced TEST on that concept
        result = self.stack.service.next_action(COURSE)
        test_candidates = [c for c in result.candidates if c.mode is NextMode.TEST]
        self.assertTrue(any(c.concept_id == "cyber/cia" for c in test_candidates))

    def test_state_reports_burndown(self) -> None:
        self.stack.service.grade(concept_id="cyber/cia", score=6)
        snap = self.stack.service.state(COURSE)
        self.assertEqual(snap["total"], 3)
        self.assertEqual(snap["by_mastery"]["solid"], 1)
        self.assertEqual(snap["by_mastery"]["new"], 2)


if __name__ == "__main__":
    unittest.main()

"""Tests for the honest snapshot metrics (engine.snapshot).

Each function is a pure reading of the current learner state -- no clock reads,
no I/O -- so these tests build tiny states and assert the named invariants:

* stability_days weights every state's stability by its concept importance
  (importance padding, not raw corpus size, is what moves the number),
* ripeness buckets due reviews by CALENDAR DAY into gain-framed names, with
  already-passed due dates folded into ready_now by design,
* unlock_proximity surfaces never-seen concepts that are one mastered
  prerequisite away from unlocking,
* consolidation_report counts intervals still intact and reviews done since a
  reference instant.

``now`` (and ``since``) are always injected, so every assertion is
deterministic.
"""
from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from curriculum.domain.entities import Concept, Edge, LearnerState
from curriculum.domain.enums import EdgeType, Mastery
from curriculum.engine.snapshot import (
    consolidation_report,
    ripeness,
    stability_days,
    unlock_proximity,
)

NOW = datetime(2026, 7, 3, 8, 0, tzinfo=timezone.utc)


def _concept(cid: str, importance: float = 0.5) -> Concept:
    return Concept(id=cid, course="sec", title=cid, importance=importance)


def _prereq_edge(prereq: str, dst: str) -> Edge:
    return Edge(src=prereq, dst=dst, type=EdgeType.PREREQUISITE)


# --------------------------------------------------------------------------- #
# stability_days
# --------------------------------------------------------------------------- #
class StabilityDaysTests(unittest.TestCase):
    def test_importance_weights_the_sum(self):
        # importance 1.0 * stability 10 + importance 0.5 * stability 20 == 20.0
        concepts = {"a": _concept("a", 1.0), "b": _concept("b", 0.5)}
        states = [
            LearnerState(concept_id="a", stability=10.0),
            LearnerState(concept_id="b", stability=20.0),
        ]
        self.assertEqual(stability_days(states, concepts), 20.0)

    def test_never_seen_states_are_excluded(self):
        # a None-stability (never seen) state contributes nothing.
        concepts = {"a": _concept("a", 1.0), "b": _concept("b", 1.0)}
        states = [
            LearnerState(concept_id="a", stability=10.0),
            LearnerState(concept_id="b", stability=None),
        ]
        self.assertEqual(stability_days(states, concepts), 10.0)

    def test_state_absent_from_concepts_is_skipped(self):
        concepts = {"a": _concept("a", 1.0)}
        states = [
            LearnerState(concept_id="a", stability=10.0),
            LearnerState(concept_id="ghost", stability=99.0),
        ]
        self.assertEqual(stability_days(states, concepts), 10.0)

    def test_empty_states_is_zero(self):
        self.assertEqual(stability_days([], {}), 0.0)


# --------------------------------------------------------------------------- #
# ripeness
# --------------------------------------------------------------------------- #
class RipenessTests(unittest.TestCase):
    def _state(self, cid: str, due: datetime) -> LearnerState:
        return LearnerState(concept_id=cid, stability=5.0, due_at=due)

    def test_bucket_edges(self):
        states = [
            self._state("yesterday", NOW - timedelta(days=1)),
            self._state("today", NOW.replace(hour=23)),  # later today, still ready_now
            self._state("tomorrow", NOW + timedelta(days=1)),
            self._state("in_three", NOW + timedelta(days=3)),
            self._state("in_thirty", NOW + timedelta(days=30)),
        ]
        out = ripeness(states, NOW)
        self.assertEqual(out["ready_now"], ["today", "yesterday"])
        self.assertEqual(out["ready_tomorrow"], ["tomorrow"])
        self.assertEqual(out["ready_this_week"], ["in_three"])
        self.assertEqual(out["holding"], ["in_thirty"])

    def test_due_earlier_today_is_ready_now(self):
        # a review due at 08:00 is ready when now is 23:00 the same day.
        late = NOW.replace(hour=23)
        state = LearnerState(concept_id="c", stability=5.0, due_at=NOW)
        out = ripeness([state], late)
        self.assertEqual(out["ready_now"], ["c"])

    def test_week_edge_is_inclusive_seven_days(self):
        states = [self._state("edge", NOW + timedelta(days=7))]
        out = ripeness(states, NOW)
        self.assertEqual(out["ready_this_week"], ["edge"])

    def test_states_without_due_are_ignored(self):
        state = LearnerState(concept_id="c", stability=None, due_at=None)
        out = ripeness([state], NOW)
        self.assertEqual(
            out,
            {
                "ready_now": [],
                "ready_tomorrow": [],
                "ready_this_week": [],
                "holding": [],
            },
        )

    def test_ready_now_is_sorted(self):
        states = [
            self._state("zeta", NOW - timedelta(days=1)),
            self._state("alpha", NOW - timedelta(days=1)),
        ]
        out = ripeness(states, NOW)
        self.assertEqual(out["ready_now"], ["alpha", "zeta"])


# --------------------------------------------------------------------------- #
# unlock_proximity
# --------------------------------------------------------------------------- #
class UnlockProximityTests(unittest.TestCase):
    def test_one_away_when_single_unmastered_prereq_is_learning(self):
        target = _concept("t")
        states = {
            "p_solid": LearnerState(concept_id="p_solid", stability=9.0, mastery=Mastery.SOLID),
            "p_learning": LearnerState(
                concept_id="p_learning", stability=2.0, mastery=Mastery.LEARNING
            ),
        }
        prereq_in_edges = {
            "t": [_prereq_edge("p_solid", "t"), _prereq_edge("p_learning", "t")]
        }
        out = unlock_proximity([target], states, prereq_in_edges)
        self.assertEqual(out, [{"concept_id": "t", "missing": 1, "one_away": True}])

    def test_not_one_away_when_missing_prereq_is_new(self):
        target = _concept("t")
        states = {
            "p_solid": LearnerState(concept_id="p_solid", stability=9.0, mastery=Mastery.SOLID),
            "p_new": LearnerState(concept_id="p_new", stability=None, mastery=Mastery.NEW),
        }
        prereq_in_edges = {
            "t": [_prereq_edge("p_solid", "t"), _prereq_edge("p_new", "t")]
        }
        out = unlock_proximity([target], states, prereq_in_edges)
        self.assertEqual(out, [{"concept_id": "t", "missing": 1, "one_away": False}])

    def test_prereq_with_no_state_counts_as_unmastered_and_not_one_away(self):
        target = _concept("t")
        prereq_in_edges = {"t": [_prereq_edge("p_unknown", "t")]}
        out = unlock_proximity([target], {}, prereq_in_edges)
        self.assertEqual(out, [{"concept_id": "t", "missing": 1, "one_away": False}])

    def test_fully_unlocked_concept_is_omitted(self):
        target = _concept("t")
        states = {
            "p": LearnerState(concept_id="p", stability=9.0, mastery=Mastery.EXAM_READY)
        }
        prereq_in_edges = {"t": [_prereq_edge("p", "t")]}
        out = unlock_proximity([target], states, prereq_in_edges)
        self.assertEqual(out, [])

    def test_already_seen_concept_is_omitted(self):
        target = _concept("t")
        states = {
            "t": LearnerState(concept_id="t", stability=3.0, mastery=Mastery.LEARNING),
            "p": LearnerState(concept_id="p", stability=None, mastery=Mastery.NEW),
        }
        prereq_in_edges = {"t": [_prereq_edge("p", "t")]}
        out = unlock_proximity([target], states, prereq_in_edges)
        self.assertEqual(out, [])

    def test_sorted_by_missing_then_concept_id(self):
        c_a = _concept("a")  # 2 missing
        c_b = _concept("b")  # 1 missing
        c_c = _concept("c")  # 1 missing
        states = {}
        prereq_in_edges = {
            "a": [_prereq_edge("x", "a"), _prereq_edge("y", "a")],
            "b": [_prereq_edge("x", "b")],
            "c": [_prereq_edge("x", "c")],
        }
        out = unlock_proximity([c_a, c_b, c_c], states, prereq_in_edges)
        self.assertEqual(
            [(r["concept_id"], r["missing"]) for r in out],
            [("b", 1), ("c", 1), ("a", 2)],
        )


# --------------------------------------------------------------------------- #
# consolidation_report
# --------------------------------------------------------------------------- #
class ConsolidationReportTests(unittest.TestCase):
    def test_holding_counts_intervals_still_intact(self):
        states = [
            LearnerState(concept_id="held", stability=5.0, due_at=NOW + timedelta(days=2)),
            LearnerState(concept_id="due", stability=5.0, due_at=NOW - timedelta(days=1)),
            LearnerState(concept_id="unscheduled", stability=None, due_at=None),
        ]
        out = consolidation_report(states, since=None, now=NOW)
        self.assertEqual(out["holding"], 1)

    def test_reviewed_since_counts_reviews_at_or_after_since(self):
        since = NOW - timedelta(days=7)
        states = [
            LearnerState(concept_id="recent", stability=5.0, last_review=NOW - timedelta(days=1)),
            LearnerState(concept_id="on_edge", stability=5.0, last_review=since),
            LearnerState(concept_id="old", stability=5.0, last_review=NOW - timedelta(days=30)),
            LearnerState(concept_id="never", stability=None, last_review=None),
        ]
        out = consolidation_report(states, since=since, now=NOW)
        self.assertEqual(out["reviewed_since"], 2)

    def test_since_none_yields_zero_reviewed_since(self):
        states = [
            LearnerState(concept_id="recent", stability=5.0, last_review=NOW - timedelta(days=1))
        ]
        out = consolidation_report(states, since=None, now=NOW)
        self.assertEqual(out["reviewed_since"], 0)

    def test_empty_states(self):
        out = consolidation_report([], since=NOW, now=NOW)
        self.assertEqual(out, {"holding": 0, "reviewed_since": 0})


# --------------------------------------------------------------------------- #
# determinism
# --------------------------------------------------------------------------- #
class DeterminismTests(unittest.TestCase):
    def test_identical_inputs_give_identical_output(self):
        concepts = {"a": _concept("a", 1.0)}
        states = [LearnerState(concept_id="a", stability=10.0, due_at=NOW + timedelta(days=1))]
        self.assertEqual(stability_days(states, concepts), stability_days(states, concepts))
        self.assertEqual(ripeness(states, NOW), ripeness(states, NOW))
        self.assertEqual(
            consolidation_report(states, None, NOW),
            consolidation_report(states, None, NOW),
        )


if __name__ == "__main__":
    unittest.main()

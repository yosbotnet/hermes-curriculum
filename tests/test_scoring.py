"""Tests for the selection scoring terms (engine.scoring).

Each term is a pure value function over a CandidateContext, so these tests build
tiny contexts and assert the named invariants directly:

* urgency rises as retrievability falls (and never-seen is moderate),
* difficulty_fit peaks at ~0.85 and falls on either side,
* exploration decreases monotonically with visits,
* interleave_penalty fires ONLY for a same-cluster item after basics,
* coverage rises with staleness and as the exam nears.

No clock or RNG is involved -- ``now`` is injected via the context -- so every
assertion is deterministic.
"""
from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from curriculum.domain.entities import (
    CandidateContext,
    Concept,
    CourseProfile,
    LearnerState,
)
from curriculum.domain.enums import Mastery, NextMode
from curriculum.ports.strategies import ScoringTerm
from curriculum.engine.scoring import (
    CoverageTerm,
    DifficultyFitTerm,
    ExplorationTerm,
    InterleavePenaltyTerm,
    UrgencyTerm,
)

NOW = datetime(2026, 6, 30, 12, 0, tzinfo=timezone.utc)
CONCEPT = Concept(id="c1", course="sec", title="Concept 1")


def make_ctx(
    *,
    retrievability=None,
    state=None,
    cluster=None,
    visits=0,
    days_to_exam=None,
    target_retention=0.90,
    mode=NextMode.REVIEW,
    extra=None,
    now=NOW,
) -> CandidateContext:
    """Build a CandidateContext with sensible defaults, overriding only the
    field(s) a given test cares about. Keeps each test focused on one lever."""
    profile = CourseProfile(course="sec", archetype="conceptual", target_retention=target_retention)
    return CandidateContext(
        concept=CONCEPT,
        mode=mode,
        state=state,
        retrievability=retrievability,
        now=now,
        profile=profile,
        cluster=cluster,
        visits=visits,
        days_to_exam=days_to_exam,
        extra=extra if extra is not None else {},
    )


def state_with(*, mastery=Mastery.LEARNING, last_review=None) -> LearnerState:
    """A LearnerState carrying only the fields the scoring terms read."""
    return LearnerState(
        concept_id="c1",
        stability=10.0,
        difficulty=5.0,
        last_review=last_review,
        mastery=mastery,
    )


class CommonContractTests(unittest.TestCase):
    """Every term must honour the ScoringTerm port: right name, is-a, float."""

    ALL = [
        (UrgencyTerm, "urgency"),
        (DifficultyFitTerm, "difficulty_fit"),
        (ExplorationTerm, "exploration"),
        (InterleavePenaltyTerm, "interleave_penalty"),
        (CoverageTerm, "coverage"),
    ]

    def test_names_match_spec(self):
        for cls, expected in self.ALL:
            self.assertEqual(cls.name, expected)

    def test_all_are_scoring_terms(self):
        for cls, _ in self.ALL:
            self.assertIsInstance(cls(), ScoringTerm)

    def test_all_return_float(self):
        ctx = make_ctx(retrievability=0.5, state=state_with(), days_to_exam=10)
        for cls, _ in self.ALL:
            self.assertIsInstance(cls().score(ctx), float)


class UrgencyTermTests(unittest.TestCase):
    def setUp(self):
        self.term = UrgencyTerm()

    def test_never_seen_is_moderate(self):
        self.assertEqual(self.term.score(make_ctx(retrievability=None)), 0.5)

    def test_rises_as_retrievability_falls(self):
        high_r = self.term.score(make_ctx(retrievability=0.85))
        mid_r = self.term.score(make_ctx(retrievability=0.5))
        low_r = self.term.score(make_ctx(retrievability=0.1))
        self.assertLess(high_r, mid_r)
        self.assertLess(mid_r, low_r)

    def test_at_or_above_target_is_zero(self):
        self.assertEqual(self.term.score(make_ctx(retrievability=0.90)), 0.0)
        self.assertEqual(self.term.score(make_ctx(retrievability=0.99)), 0.0)

    def test_full_miss_scales_to_one(self):
        self.assertAlmostEqual(self.term.score(make_ctx(retrievability=0.0)), 1.0)

    def test_scaling_is_relative_to_target(self):
        # Same absolute R but a lower target -> smaller debt (R is closer to it).
        at_high_target = self.term.score(make_ctx(retrievability=0.4, target_retention=0.9))
        at_low_target = self.term.score(make_ctx(retrievability=0.4, target_retention=0.5))
        self.assertGreater(at_high_target, at_low_target)

    def test_degenerate_zero_target_is_safe(self):
        self.assertEqual(self.term.score(make_ctx(retrievability=0.3, target_retention=0.0)), 0.0)

    def test_in_unit_range(self):
        for r in (0.0, 0.2, 0.5, 0.8, 1.0):
            score = self.term.score(make_ctx(retrievability=r))
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)


class DifficultyFitTermTests(unittest.TestCase):
    def setUp(self):
        self.term = DifficultyFitTerm()

    def test_peaks_at_085(self):
        self.assertAlmostEqual(self.term.score(make_ctx(retrievability=0.85)), 1.0)

    def test_falls_on_both_sides_of_peak(self):
        peak = self.term.score(make_ctx(retrievability=0.85))
        below = self.term.score(make_ctx(retrievability=0.60))
        above = self.term.score(make_ctx(retrievability=0.98))
        self.assertGreater(peak, below)
        self.assertGreater(peak, above)

    def test_monotone_approaching_peak_from_below(self):
        s_low = self.term.score(make_ctx(retrievability=0.3))
        s_mid = self.term.score(make_ctx(retrievability=0.6))
        s_near = self.term.score(make_ctx(retrievability=0.8))
        self.assertLess(s_low, s_mid)
        self.assertLess(s_mid, s_near)

    def test_none_uses_prior(self):
        # p == 0.5 prior -> 1 - (0.35 / 0.85)
        expected = 1.0 - (abs(0.5 - 0.85) / 0.85)
        self.assertAlmostEqual(self.term.score(make_ctx(retrievability=None)), expected)

    def test_clamped_non_negative_at_zero_recall(self):
        # p == 0 -> 1 - 0.85/0.85 == 0.0 exactly, never below.
        self.assertAlmostEqual(self.term.score(make_ctx(retrievability=0.0)), 0.0)

    def test_in_unit_range(self):
        for r in (0.0, 0.25, 0.5, 0.85, 1.0):
            score = self.term.score(make_ctx(retrievability=r))
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)


class ExplorationTermTests(unittest.TestCase):
    def setUp(self):
        self.term = ExplorationTerm()

    def test_unvisited_is_one(self):
        self.assertEqual(self.term.score(make_ctx(visits=0)), 1.0)

    def test_decreases_with_visits(self):
        scores = [self.term.score(make_ctx(visits=v)) for v in (0, 1, 3, 10, 50)]
        for earlier, later in zip(scores, scores[1:]):
            self.assertGreater(earlier, later)

    def test_known_values(self):
        self.assertAlmostEqual(self.term.score(make_ctx(visits=1)), 0.5)
        self.assertAlmostEqual(self.term.score(make_ctx(visits=3)), 0.25)

    def test_negative_visits_clamped(self):
        # A malformed negative count must not exceed the unvisited bonus.
        self.assertEqual(self.term.score(make_ctx(visits=-5)), 1.0)


class InterleavePenaltyTermTests(unittest.TestCase):
    def setUp(self):
        self.term = InterleavePenaltyTerm()

    def test_fires_for_same_cluster_after_basics(self):
        ctx = make_ctx(
            cluster="crypto",
            state=state_with(mastery=Mastery.SOLID),
            extra={"last_cluster": "crypto"},
        )
        self.assertGreater(self.term.score(ctx), 0.0)

    def test_no_penalty_when_still_new(self):
        ctx = make_ctx(
            cluster="crypto",
            state=state_with(mastery=Mastery.NEW),
            extra={"last_cluster": "crypto"},
        )
        self.assertEqual(self.term.score(ctx), 0.0)

    def test_no_penalty_when_state_missing(self):
        ctx = make_ctx(cluster="crypto", state=None, extra={"last_cluster": "crypto"})
        self.assertEqual(self.term.score(ctx), 0.0)

    def test_no_penalty_for_different_cluster(self):
        ctx = make_ctx(
            cluster="crypto",
            state=state_with(mastery=Mastery.SOLID),
            extra={"last_cluster": "networks"},
        )
        self.assertEqual(self.term.score(ctx), 0.0)

    def test_no_penalty_when_candidate_has_no_cluster(self):
        # cluster None must not match a None last_cluster and trigger a penalty.
        ctx = make_ctx(cluster=None, state=state_with(mastery=Mastery.SOLID), extra={})
        self.assertEqual(self.term.score(ctx), 0.0)

    def test_no_penalty_when_last_cluster_absent(self):
        ctx = make_ctx(cluster="crypto", state=state_with(mastery=Mastery.SOLID), extra={})
        self.assertEqual(self.term.score(ctx), 0.0)

    def test_penalty_is_positive_magnitude(self):
        ctx = make_ctx(
            cluster="crypto",
            state=state_with(mastery=Mastery.EXAM_READY),
            extra={"last_cluster": "crypto"},
        )
        self.assertGreaterEqual(self.term.score(ctx), 0.0)


class CoverageTermTests(unittest.TestCase):
    def setUp(self):
        self.term = CoverageTerm()

    def test_rises_with_staleness(self):
        recent = make_ctx(state=state_with(last_review=NOW - timedelta(days=1)), days_to_exam=30)
        old = make_ctx(state=state_with(last_review=NOW - timedelta(days=60)), days_to_exam=30)
        self.assertLess(self.term.score(recent), self.term.score(old))

    def test_never_seen_is_max_staleness(self):
        never = make_ctx(state=None, days_to_exam=30)
        seen_recent = make_ctx(state=state_with(last_review=NOW - timedelta(days=2)), days_to_exam=30)
        self.assertGreater(self.term.score(never), self.term.score(seen_recent))

    def test_rises_as_exam_nears(self):
        far = make_ctx(state=state_with(last_review=NOW - timedelta(days=10)), days_to_exam=120)
        near = make_ctx(state=state_with(last_review=NOW - timedelta(days=10)), days_to_exam=3)
        self.assertLess(self.term.score(far), self.term.score(near))

    def test_no_exam_has_no_deadline_pressure(self):
        # With no exam date, two same-staleness items differ only by deadline:
        # the one with an imminent exam must score strictly higher.
        no_exam = make_ctx(state=state_with(last_review=NOW - timedelta(days=10)), days_to_exam=None)
        imminent = make_ctx(state=state_with(last_review=NOW - timedelta(days=10)), days_to_exam=1)
        self.assertLess(self.term.score(no_exam), self.term.score(imminent))

    def test_past_due_exam_is_max_pressure(self):
        # A negative days_to_exam (exam already passed) must not reduce pressure.
        on_day = make_ctx(state=state_with(last_review=NOW - timedelta(days=5)), days_to_exam=0)
        past = make_ctx(state=state_with(last_review=NOW - timedelta(days=5)), days_to_exam=-10)
        self.assertAlmostEqual(self.term.score(on_day), self.term.score(past))

    def test_clock_skew_does_not_break_monotonicity(self):
        # now < last_review (skew): staleness floored at 0, never negative.
        future_review = make_ctx(state=state_with(last_review=NOW + timedelta(days=5)), days_to_exam=30)
        at_now = make_ctx(state=state_with(last_review=NOW), days_to_exam=30)
        self.assertAlmostEqual(self.term.score(future_review), self.term.score(at_now))

    def test_in_unit_range(self):
        for days_seen, dte in ((0, 0), (10, 30), (100, 5), (1, 365)):
            ctx = make_ctx(
                state=state_with(last_review=NOW - timedelta(days=days_seen)),
                days_to_exam=dte,
            )
            score = self.term.score(ctx)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()

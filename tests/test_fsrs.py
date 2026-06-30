"""Tests for the FSRS-style DSR scheduler.

Stdlib unittest only. Determinism comes for free: the scheduler is pure and the
clock is injected, so every assertion is reproducible. We pin a fixed ``now``
and build states by chaining reviews exactly as the application would.
"""
from __future__ import annotations

import math
import unittest
from datetime import datetime, timedelta

from curriculum.domain.entities import LearnerState
from curriculum.domain.enums import FsrsRating, Mastery
from curriculum.engine.fsrs import DECAY, FACTOR, W, _S_MIN, FsrsScheduler

T0 = datetime(2026, 1, 1, 12, 0, 0)


class ConstantsTest(unittest.TestCase):
    def test_version_is_pinned(self) -> None:
        self.assertEqual(FsrsScheduler().version, "fsrs-v1")

    def test_forgetting_curve_constants(self) -> None:
        # DECAY/FACTOR are the FSRS-4.5 contract; FACTOR is derived so R(S,S)=0.9.
        self.assertEqual(DECAY, -0.5)
        self.assertAlmostEqual(FACTOR, 19.0 / 81.0)
        self.assertAlmostEqual(FACTOR, 0.9 ** (1.0 / DECAY) - 1.0)

    def test_published_weight_count(self) -> None:
        self.assertEqual(len(W), 19)


class RetrievabilityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sched = FsrsScheduler()

    def test_unseen_item_returns_zero(self) -> None:
        # No stability -> no trace to recall.
        self.assertEqual(
            self.sched.retrievability(LearnerState(concept_id="c"), T0), 0.0
        )
        # Stability but never reviewed -> cannot place on the curve.
        self.assertEqual(
            self.sched.retrievability(
                LearnerState(concept_id="c", stability=10.0), T0
            ),
            0.0,
        )

    def test_recall_is_one_at_t_zero(self) -> None:
        st = LearnerState(concept_id="c", stability=10.0, last_review=T0)
        self.assertAlmostEqual(self.sched.retrievability(st, T0), 1.0)

    def test_clock_skew_cannot_exceed_one(self) -> None:
        # now earlier than last_review: elapsed is clamped to 0 -> R == 1, not >1.
        st = LearnerState(concept_id="c", stability=10.0, last_review=T0)
        self.assertAlmostEqual(
            self.sched.retrievability(st, T0 - timedelta(days=5)), 1.0
        )

    def test_recall_decays_with_time(self) -> None:
        st = LearnerState(concept_id="c", stability=10.0, last_review=T0)
        r1 = self.sched.retrievability(st, T0 + timedelta(days=1))
        r30 = self.sched.retrievability(st, T0 + timedelta(days=30))
        self.assertGreater(r1, r30)
        for r in (r1, r30):
            self.assertGreaterEqual(r, 0.0)
            self.assertLessEqual(r, 1.0)

    def test_recall_equals_target_at_scheduled_interval_default(self) -> None:
        # The interval is chosen so recall has decayed to the target by due_at.
        s = self.sched.review(None, FsrsRating.GOOD, T0, target_retention=0.9)
        self.assertAlmostEqual(
            self.sched.retrievability(s, s.due_at), 0.9, places=5
        )

    def test_recall_equals_target_at_scheduled_interval_custom(self) -> None:
        s = self.sched.review(None, FsrsRating.GOOD, T0, target_retention=0.8)
        self.assertAlmostEqual(
            self.sched.retrievability(s, s.due_at), 0.8, places=5
        )


class FirstReviewTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sched = FsrsScheduler()

    def test_good_initialises_from_published_weights(self) -> None:
        out = self.sched.review(None, FsrsRating.GOOD, T0, target_retention=0.9)
        # S0(GOOD) == W[2]; D0(GOOD) == clamp(W[4]-exp(2*W[5])+1, 1, 10).
        self.assertAlmostEqual(out.stability, W[2])
        expected_d = max(1.0, min(10.0, W[4] - math.exp(W[5] * 2) + 1.0))
        self.assertAlmostEqual(out.difficulty, expected_d)
        self.assertEqual(out.reps, 1)
        self.assertEqual(out.lapses, 0)
        self.assertEqual(out.last_review, T0)
        self.assertEqual(out.mastery, Mastery.NEW)
        self.assertEqual(out.concept_id, "")
        self.assertGreater(out.due_at, T0)

    def test_each_rating_initialises_its_own_prior(self) -> None:
        for rating in FsrsRating:
            out = self.sched.review(None, rating, T0, target_retention=0.9)
            self.assertAlmostEqual(out.stability, W[rating - 1])
            self.assertGreaterEqual(out.difficulty, 1.0)
            self.assertLessEqual(out.difficulty, 10.0)

    def test_first_again_counts_as_a_lapse(self) -> None:
        out = self.sched.review(None, FsrsRating.AGAIN, T0, target_retention=0.9)
        self.assertAlmostEqual(out.stability, W[0])
        self.assertEqual(out.lapses, 1)
        self.assertEqual(out.reps, 1)

    def test_state_without_prior_stability_uses_first_review_path(self) -> None:
        # A state object exists (carries identity) but was never reviewed.
        seed = LearnerState(concept_id="abc", reps=0, mastery=Mastery.LEARNING)
        out = self.sched.review(seed, FsrsRating.GOOD, T0, target_retention=0.9)
        self.assertAlmostEqual(out.stability, W[2])
        self.assertEqual(out.concept_id, "abc")
        self.assertEqual(out.reps, 1)
        self.assertEqual(out.mastery, Mastery.LEARNING)  # preserved


class StabilityDynamicsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sched = FsrsScheduler()
        # A first GOOD review gives a known starting stability (W[2]) and a
        # due_at exactly one stability out (target 0.9 => interval == S).
        self.s1 = self.sched.review(None, FsrsRating.GOOD, T0, target_retention=0.9)

    def test_stability_grows_after_good(self) -> None:
        s2 = self.sched.review(
            self.s1, FsrsRating.GOOD, self.s1.due_at, target_retention=0.9
        )
        self.assertGreater(s2.stability, self.s1.stability)

    def test_easy_grows_stability_more_than_good(self) -> None:
        good = self.sched.review(
            self.s1, FsrsRating.GOOD, self.s1.due_at, target_retention=0.9
        )
        easy = self.sched.review(
            self.s1, FsrsRating.EASY, self.s1.due_at, target_retention=0.9
        )
        self.assertGreater(easy.stability, good.stability)

    def test_spacing_effect_late_review_grows_more(self) -> None:
        # Recalling at lower retrievability (more overdue) yields a bigger jump.
        early = self.sched.review(
            self.s1,
            FsrsRating.GOOD,
            self.s1.last_review + timedelta(days=1),
            target_retention=0.9,
        )
        late = self.sched.review(
            self.s1,
            FsrsRating.GOOD,
            self.s1.last_review + timedelta(days=6),
            target_retention=0.9,
        )
        self.assertGreater(late.stability, early.stability)

    def test_again_lowers_stability_and_increments_lapses(self) -> None:
        lapsed = self.sched.review(
            self.s1, FsrsRating.AGAIN, self.s1.due_at, target_retention=0.9
        )
        self.assertLess(lapsed.stability, self.s1.stability)
        self.assertEqual(lapsed.lapses, self.s1.lapses + 1)
        self.assertEqual(lapsed.reps, self.s1.reps + 1)

    def test_stability_never_drops_below_floor(self) -> None:
        state = None
        now = T0
        for _ in range(30):
            state = self.sched.review(
                state, FsrsRating.AGAIN, now, target_retention=0.9
            )
            self.assertGreaterEqual(state.stability, _S_MIN)
            now += timedelta(days=1)


class DifficultyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sched = FsrsScheduler()

    def test_difficulty_stays_in_range_over_many_reviews(self) -> None:
        state = None
        now = T0
        # Hammer toward the hard ceiling, then toward the easy floor; D must
        # remain inside [1, 10] at every step.
        for rating in [FsrsRating.AGAIN] * 25 + [FsrsRating.EASY] * 25:
            state = self.sched.review(state, rating, now, target_retention=0.9)
            self.assertGreaterEqual(state.difficulty, 1.0)
            self.assertLessEqual(state.difficulty, 10.0)
            now += timedelta(days=1)

    def test_again_raises_difficulty_easy_lowers_it(self) -> None:
        base = self.sched.review(None, FsrsRating.GOOD, T0, target_retention=0.9)
        harder = self.sched.review(
            base, FsrsRating.AGAIN, base.due_at, target_retention=0.9
        )
        easier = self.sched.review(
            base, FsrsRating.EASY, base.due_at, target_retention=0.9
        )
        self.assertGreater(harder.difficulty, base.difficulty)
        self.assertLess(easier.difficulty, base.difficulty)


class IntervalTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sched = FsrsScheduler()

    @staticmethod
    def _interval_days(state: LearnerState) -> float:
        return (state.due_at - state.last_review).total_seconds() / 86400.0

    def test_interval_equals_stability_at_default_retention(self) -> None:
        s = self.sched.review(None, FsrsRating.GOOD, T0, target_retention=0.9)
        self.assertAlmostEqual(self._interval_days(s), s.stability, places=6)

    def test_interval_grows_with_stability(self) -> None:
        s1 = self.sched.review(None, FsrsRating.GOOD, T0, target_retention=0.9)
        s2 = self.sched.review(s1, FsrsRating.GOOD, s1.due_at, target_retention=0.9)
        self.assertGreater(s2.stability, s1.stability)
        self.assertGreater(self._interval_days(s2), self._interval_days(s1))

    def test_higher_target_retention_shortens_interval(self) -> None:
        # Demanding higher recall forces an earlier review for the same memory.
        loose = self.sched.review(None, FsrsRating.GOOD, T0, target_retention=0.80)
        tight = self.sched.review(None, FsrsRating.GOOD, T0, target_retention=0.95)
        self.assertEqual(loose.stability, tight.stability)
        self.assertGreater(self._interval_days(loose), self._interval_days(tight))


class PreservationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sched = FsrsScheduler()

    def test_mastery_and_identity_are_preserved(self) -> None:
        seed = LearnerState(
            concept_id="cid-1",
            stability=5.0,
            difficulty=5.0,
            last_review=T0,
            due_at=T0,
            reps=3,
            lapses=1,
            mastery=Mastery.SOLID,
        )
        out = self.sched.review(
            seed, FsrsRating.GOOD, T0 + timedelta(days=5), target_retention=0.9
        )
        self.assertEqual(out.mastery, Mastery.SOLID)  # application owns mastery
        self.assertEqual(out.concept_id, "cid-1")
        self.assertEqual(out.reps, 4)
        self.assertEqual(out.lapses, 1)  # GOOD is not a lapse
        self.assertEqual(out.last_review, T0 + timedelta(days=5))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

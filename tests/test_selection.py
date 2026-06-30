"""Tests for the WeightedSamplingPolicy (engine.selection).

The policy is deterministic given its inputs: randomness flows through an
injected ``random.Random`` and time through the candidate contexts. So every
test seeds the RNG (or drives temperature to the argmax floor) and asserts the
named invariants directly:

* near the exam (small days_to_exam) the softmax collapses to ~argmax,
* a hard-due candidate always wins, bypassing the lottery,
* a seeded sample is reproducible,
* the ranked field is sorted strictly descending by score,
* per-term weights from EngineConfig are applied,
* a same-cluster-after-basics candidate is scored LOWER via the interleave
  penalty (the penalty is subtracted),
* empty candidates raise NoCandidatesAvailable.

Custom ``ScoringTerm`` fakes keep the scores fully controllable so the sampling
and ranking behaviour can be asserted without reverse-engineering the real
term math; one test also exercises the real InterleavePenaltyTerm end to end.
"""
from __future__ import annotations

import random
import unittest
from datetime import datetime, timezone

from curriculum.domain import errors
from curriculum.domain.entities import (
    CandidateContext,
    Concept,
    CourseProfile,
    EngineConfig,
    LearnerState,
    NextResult,
    ScoredCandidate,
    SourceRef,
)
from curriculum.domain.enums import Mastery, NextMode
from curriculum.engine.scoring import InterleavePenaltyTerm
from curriculum.engine.selection import _TEMP_EPSILON, WeightedSamplingPolicy
from curriculum.ports.strategies import ScoringTerm

NOW = datetime(2026, 6, 30, 12, 0, tzinfo=timezone.utc)
PROFILE = CourseProfile(course="sec", archetype="conceptual")


class ExtraScoreTerm(ScoringTerm):
    """Fake term: returns whatever score the candidate carries in ``extra``.

    Lets a test set each candidate's contribution explicitly via
    ``extra={'score': x}``, so the policy's composition/sampling can be checked
    against known numbers rather than the real terms' heuristics.
    """

    name = "extra_score"

    def score(self, ctx: CandidateContext) -> float:
        return float(ctx.extra.get("score", 0.0))


class ConstTerm(ScoringTerm):
    """Fake term that always returns a fixed value under a chosen name.

    Used to verify per-term weighting: two ConstTerms with different names can
    be weighted independently through EngineConfig.weights.
    """

    def __init__(self, name: str, value: float) -> None:
        self.name = name
        self._value = value

    def score(self, ctx: CandidateContext) -> float:
        return self._value


def make_candidate(
    concept_id: str,
    *,
    score: float = 0.0,
    mode: NextMode = NextMode.REVIEW,
    days_to_exam: int | None = None,
    hard_due: bool = False,
    cluster: str | None = None,
    state: LearnerState | None = None,
    source_refs: tuple[SourceRef, ...] = (),
    extra: dict | None = None,
) -> CandidateContext:
    """Build a CandidateContext, defaulting everything the policy ignores.

    ``score`` is stashed in ``extra`` for ExtraScoreTerm to read back, so the
    candidate's composed score is controllable from a single argument.
    """
    payload = {"score": score}
    if extra:
        payload.update(extra)
    concept = Concept(id=concept_id, course="sec", title=concept_id, source_refs=source_refs)
    return CandidateContext(
        concept=concept,
        mode=mode,
        state=state,
        retrievability=None,
        now=NOW,
        profile=PROFILE,
        cluster=cluster,
        visits=0,
        days_to_exam=days_to_exam,
        hard_due=hard_due,
        extra=payload,
    )


def policy(rng_seed: int | None = None, **kwargs) -> WeightedSamplingPolicy:
    """A policy wired with the controllable ExtraScoreTerm by default."""
    rng = random.Random(rng_seed) if rng_seed is not None else None
    terms = kwargs.pop("terms", [ExtraScoreTerm()])
    return WeightedSamplingPolicy(terms, rng=rng, **kwargs)


def cfg(**kwargs) -> EngineConfig:
    return EngineConfig(**kwargs)


class EmptyCandidatesTests(unittest.TestCase):
    def test_empty_raises_no_candidates(self):
        with self.assertRaises(errors.NoCandidatesAvailable):
            policy(rng_seed=1).select([], config=cfg(), now=NOW)


class RankingTests(unittest.TestCase):
    def test_ranked_sorted_descending(self):
        cands = [
            make_candidate("a", score=0.2),
            make_candidate("b", score=0.9),
            make_candidate("c", score=0.5),
            make_candidate("d", score=0.7),
        ]
        result = policy(rng_seed=0).select(cands, config=cfg(), now=NOW)
        scores = [sc.score for sc in result.candidates]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual([sc.concept_id for sc in result.candidates], ["b", "d", "c", "a"])

    def test_ranked_carries_every_candidate(self):
        cands = [make_candidate(cid, score=float(i)) for i, cid in enumerate("abcde")]
        result = policy(rng_seed=0).select(cands, config=cfg(), now=NOW)
        self.assertEqual(len(result.candidates), len(cands))
        self.assertEqual(
            {sc.concept_id for sc in result.candidates}, {c.concept.id for c in cands}
        )

    def test_scored_candidate_carries_mode(self):
        cands = [make_candidate("a", score=1.0, mode=NextMode.TEACH)]
        result = policy(rng_seed=0).select(cands, config=cfg(), now=NOW)
        self.assertEqual(result.candidates[0].mode, NextMode.TEACH)


class WeightingTests(unittest.TestCase):
    def test_default_weight_applied_when_absent(self):
        # extra_score returns 0.4; with default_weight 2.0 and no override the
        # composed score is 0.8.
        cands = [make_candidate("a", score=0.4)]
        result = policy(rng_seed=0, default_weight=2.0).select(cands, config=cfg(), now=NOW)
        self.assertAlmostEqual(result.candidates[0].score, 0.8)

    def test_config_weight_overrides_default(self):
        # Two const terms; weight one of them and leave the other at default 1.0.
        terms = [ConstTerm("alpha", 1.0), ConstTerm("beta", 1.0)]
        pol = WeightedSamplingPolicy(terms, rng=random.Random(0))
        config = cfg(weights={"alpha": 3.0})
        result = pol.select([make_candidate("a")], config=config, now=NOW)
        # 3.0 * 1.0 (alpha) + 1.0 * 1.0 (beta default) == 4.0
        self.assertAlmostEqual(result.candidates[0].score, 4.0)

    def test_weight_changes_ranking(self):
        # Candidate X wins on alpha, candidate Y wins on beta; the heavier weight
        # decides who ranks first.
        class FieldTerm(ScoringTerm):
            def __init__(self, name):
                self.name = name

            def score(self, ctx):
                return float(ctx.extra.get(self.name, 0.0))

        terms = [FieldTerm("alpha"), FieldTerm("beta")]
        x = make_candidate("x", extra={"alpha": 1.0, "beta": 0.0})
        y = make_candidate("y", extra={"alpha": 0.0, "beta": 1.0})
        pol = WeightedSamplingPolicy(terms, rng=random.Random(0))
        # Heavier beta -> y on top.
        r1 = pol.select([x, y], config=cfg(weights={"beta": 5.0}), now=NOW)
        self.assertEqual(r1.candidates[0].concept_id, "y")
        # Heavier alpha -> x on top.
        r2 = pol.select([x, y], config=cfg(weights={"alpha": 5.0}), now=NOW)
        self.assertEqual(r2.candidates[0].concept_id, "x")


class InterleavePenaltyTests(unittest.TestCase):
    def test_same_cluster_after_basics_scored_lower(self):
        # Two candidates with identical base reward; the one in the same cluster
        # as the last pick, and past NEW, must rank lower because the interleave
        # penalty is SUBTRACTED.
        terms = [ExtraScoreTerm(), InterleavePenaltyTerm()]
        solid = LearnerState(concept_id="pen", mastery=Mastery.SOLID)
        penalised = make_candidate(
            "pen",
            score=0.5,
            cluster="crypto",
            state=solid,
            extra={"last_cluster": "crypto"},
        )
        clean = make_candidate(
            "clean",
            score=0.5,
            cluster="networks",
            state=solid,
            extra={"last_cluster": "crypto"},
        )
        pol = WeightedSamplingPolicy(terms, rng=random.Random(0))
        result = pol.select([penalised, clean], config=cfg(), now=NOW)
        by_id = {sc.concept_id: sc.score for sc in result.candidates}
        self.assertLess(by_id["pen"], by_id["clean"])
        # The clean one wins the top rank.
        self.assertEqual(result.candidates[0].concept_id, "clean")

    def test_penalty_weight_scales_subtraction(self):
        # A heavier interleave weight subtracts more, pushing the score lower.
        terms = [ExtraScoreTerm(), InterleavePenaltyTerm()]
        solid = LearnerState(concept_id="pen", mastery=Mastery.SOLID)
        cand = make_candidate(
            "pen", score=1.0, cluster="crypto", state=solid, extra={"last_cluster": "crypto"}
        )
        pol = WeightedSamplingPolicy(terms, rng=random.Random(0))
        light = pol.select([cand], config=cfg(weights={"interleave_penalty": 0.5}), now=NOW)
        heavy = pol.select([cand], config=cfg(weights={"interleave_penalty": 2.0}), now=NOW)
        self.assertGreater(light.candidates[0].score, heavy.candidates[0].score)
        # base 1.0 - 0.5*1.0 == 0.5 ; base 1.0 - 2.0*1.0 == -1.0
        self.assertAlmostEqual(light.candidates[0].score, 0.5)
        self.assertAlmostEqual(heavy.candidates[0].score, -1.0)


class TemperatureTests(unittest.TestCase):
    def test_no_deadline_uses_base_temperature(self):
        result = policy(rng_seed=0).select(
            [make_candidate("a", days_to_exam=None)], config=cfg(base_temperature=0.6), now=NOW
        )
        self.assertAlmostEqual(result.temperature, 0.6)

    def test_far_deadline_caps_at_base_temperature(self):
        # days_to_exam well beyond the horizon -> factor capped at 1.0.
        result = policy(rng_seed=0, horizon_days=30).select(
            [make_candidate("a", days_to_exam=300)], config=cfg(base_temperature=0.6), now=NOW
        )
        self.assertAlmostEqual(result.temperature, 0.6)

    def test_temperature_scales_down_near_exam(self):
        # days_to_exam == 15, horizon 30 -> factor 0.5 -> temp 0.3.
        result = policy(rng_seed=0, horizon_days=30).select(
            [make_candidate("a", days_to_exam=15)], config=cfg(base_temperature=0.6), now=NOW
        )
        self.assertAlmostEqual(result.temperature, 0.3)

    def test_temperature_floored_at_epsilon(self):
        # Exam today -> factor 0 -> temp clamped to the epsilon floor (> 0).
        result = policy(rng_seed=0).select(
            [make_candidate("a", days_to_exam=0)], config=cfg(base_temperature=0.6), now=NOW
        )
        self.assertEqual(result.temperature, _TEMP_EPSILON)
        self.assertGreater(result.temperature, 0.0)

    def test_past_due_exam_floored_at_epsilon(self):
        # Negative ratio would give a negative temp; the floor turns it into
        # argmax behaviour.
        result = policy(rng_seed=0).select(
            [make_candidate("a", days_to_exam=-10)], config=cfg(base_temperature=0.6), now=NOW
        )
        self.assertEqual(result.temperature, _TEMP_EPSILON)


class NearExamArgmaxTests(unittest.TestCase):
    def _cands(self):
        return [
            make_candidate("low", score=0.1, days_to_exam=0),
            make_candidate("top", score=0.9, days_to_exam=0),
            make_candidate("mid", score=0.5, days_to_exam=0),
        ]

    def test_near_exam_picks_argmax_regardless_of_seed(self):
        # At the epsilon temperature floor the softmax is one-hot on the best
        # score, so every seed must choose the same (top) candidate.
        for seed in range(8):
            result = policy(rng_seed=seed).select(self._cands(), config=cfg(), now=NOW)
            self.assertEqual(result.chosen.concept_id, "top")

    def test_argmax_choice_matches_top_of_ranked(self):
        result = policy(rng_seed=3).select(self._cands(), config=cfg(), now=NOW)
        self.assertEqual(result.chosen.concept_id, result.candidates[0].concept_id)


class HardDueTests(unittest.TestCase):
    def test_hard_due_wins_even_when_not_top_scorer(self):
        # The hard-due item has a LOWER score than another candidate, yet it must
        # still be chosen (the bypass ignores the sampling lottery).
        cands = [
            make_candidate("best_score", score=0.99),
            make_candidate("overdue", score=0.10, hard_due=True),
        ]
        result = policy(rng_seed=0).select(cands, config=cfg(), now=NOW)
        self.assertEqual(result.chosen.concept_id, "overdue")
        self.assertIn("hard-due", result.chosen.reason)

    def test_hard_due_picks_highest_scoring_among_overdue(self):
        cands = [
            make_candidate("over_lo", score=0.2, hard_due=True),
            make_candidate("not_due", score=0.95),
            make_candidate("over_hi", score=0.6, hard_due=True),
        ]
        result = policy(rng_seed=0).select(cands, config=cfg(), now=NOW)
        self.assertEqual(result.chosen.concept_id, "over_hi")

    def test_hard_due_is_seed_independent(self):
        cands = [
            make_candidate("a", score=0.5),
            make_candidate("due", score=0.4, hard_due=True),
            make_candidate("b", score=0.3),
        ]
        chosen = {
            policy(rng_seed=s).select(cands, config=cfg(), now=NOW).chosen.concept_id
            for s in range(6)
        }
        self.assertEqual(chosen, {"due"})

    def test_hard_due_still_returns_full_ranked_field(self):
        cands = [
            make_candidate("a", score=0.5),
            make_candidate("due", score=0.4, hard_due=True),
        ]
        result = policy(rng_seed=0).select(cands, config=cfg(), now=NOW)
        self.assertEqual(len(result.candidates), 2)
        # Ranked by score, not by due-ness: 0.5 first.
        self.assertEqual(result.candidates[0].concept_id, "a")


class SeededSamplingTests(unittest.TestCase):
    def _cands(self):
        # No deadline -> full base temperature -> real (non-argmax) sampling.
        return [make_candidate(cid, score=s, days_to_exam=None)
                for cid, s in (("a", 0.6), ("b", 0.5), ("c", 0.4), ("d", 0.3))]

    def test_same_seed_same_choice(self):
        first = policy(rng_seed=42).select(self._cands(), config=cfg(), now=NOW)
        second = policy(rng_seed=42).select(self._cands(), config=cfg(), now=NOW)
        self.assertEqual(first.chosen.concept_id, second.chosen.concept_id)

    def test_sampling_can_explore_beyond_argmax(self):
        # Over many seeds at a real temperature, the choice is not always the top
        # scorer -- proving sampling (not argmax) is in effect.
        picks = {
            policy(rng_seed=s).select(self._cands(), config=cfg(base_temperature=2.0), now=NOW)
            .chosen.concept_id
            for s in range(40)
        }
        self.assertGreater(len(picks), 1)

    def test_top_k_limits_sampling_pool(self):
        # top_k == 1 must always yield the single best candidate, whatever seed.
        cands = self._cands()
        for s in range(10):
            result = policy(rng_seed=s, top_k=1).select(
                cands, config=cfg(base_temperature=2.0), now=NOW
            )
            self.assertEqual(result.chosen.concept_id, "a")


class NextActionShapeTests(unittest.TestCase):
    def test_chosen_carries_source_refs_and_mode(self):
        refs = (SourceRef(file="notes/crypto.md", line=12),)
        cand = make_candidate(
            "c", score=1.0, mode=NextMode.TEACH, days_to_exam=0, source_refs=refs
        )
        result = policy(rng_seed=0).select([cand], config=cfg(), now=NOW)
        self.assertEqual(result.chosen.mode, NextMode.TEACH)
        self.assertEqual(result.chosen.concept_id, "c")
        self.assertEqual(result.chosen.source_refs, refs)
        self.assertIsNone(result.chosen.question_id)
        self.assertIsInstance(result.chosen.reason, str)
        self.assertTrue(result.chosen.reason)

    def test_returns_next_result_with_scored_candidates(self):
        result = policy(rng_seed=0).select(
            [make_candidate("a", score=1.0)], config=cfg(), now=NOW
        )
        self.assertIsInstance(result, NextResult)
        self.assertTrue(all(isinstance(sc, ScoredCandidate) for sc in result.candidates))


if __name__ == "__main__":
    unittest.main()

"""The default :class:`SelectionPolicy`: weighted scoring + temperature sampling.

This is the piece that turns a bag of scored candidates into ONE concrete next
action. It composes the pluggable :class:`ScoringTerm`s (Open/Closed: terms are
injected, never hard-coded here), then makes a choice that trades exploration
against exploitation through a *temperature* that decays toward an argmax as the
exam nears -- with two deterministic escape hatches that override the lottery:

* a hard-due item (something genuinely overdue) bypasses sampling entirely, so
  the schedule never gambles away a review the learner is about to forget; and
* as ``days_to_exam`` shrinks the temperature collapses toward a small epsilon,
  which makes the softmax behave like an argmax -- near the deadline we exploit
  the best-scoring item rather than explore.

Determinism is a hard requirement (see the repo rules): all randomness flows
through an injected ``random.Random`` and all time through the ``now``/context
fields, so a given seed + inputs always yields the same choice. Standard library
only.
"""
from __future__ import annotations

import math
import random
from datetime import datetime
from typing import Sequence

from ..domain import errors
from ..domain.entities import (
    CandidateContext,
    EngineConfig,
    NextAction,
    NextResult,
    ScoredCandidate,
)
from ..ports.strategies import ScoringTerm, SelectionPolicy

# The one term whose contribution is a PENALTY rather than a reward. It reports a
# positive magnitude (see InterleavePenaltyTerm), so the policy subtracts its
# weighted value instead of adding it. Named, not inlined, so the special case is
# greppable and impossible to typo silently.
_INTERLEAVE_PENALTY_NAME: str = "interleave_penalty"

# Temperature floor. A softmax at exactly zero temperature is undefined (a 0/0
# division), so we clamp to a tiny positive value. At this floor exp(delta/temp)
# is effectively a one-hot over the max-scoring item -- i.e. an argmax -- which
# is precisely the "exam is here, stop exploring" behaviour we want.
_TEMP_EPSILON: float = 1e-6


class WeightedSamplingPolicy(SelectionPolicy):
    """Compose weighted scoring terms, then temperature-sample one candidate.

    The terms, RNG and tuning knobs are all injected so the policy is pure and
    deterministic: nothing here reads a module-level clock or the global RNG.
    """

    def __init__(
        self,
        terms: Sequence[ScoringTerm],
        *,
        rng: random.Random | None = None,
        horizon_days: int = 30,
        top_k: int = 5,
        default_weight: float = 1.0,
    ) -> None:
        """Store the scoring terms and sampling knobs.

        ``terms`` is copied into a tuple so a caller mutating their list later
        cannot change this policy's behaviour (no aliasing). ``rng`` defaults to
        a fresh ``random.Random``; callers that need reproducibility pass a
        seeded instance. ``horizon_days`` is the window over which temperature
        decays as the exam nears; ``top_k`` bounds the softmax to the best few
        candidates; ``default_weight`` is used for any term absent from
        ``EngineConfig.weights``.
        """
        self._terms: tuple[ScoringTerm, ...] = tuple(terms)
        self._rng: random.Random = rng if rng is not None else random.Random()
        self._horizon_days: int = horizon_days
        self._top_k: int = top_k
        self._default_weight: float = default_weight

    # ------------------------------------------------------------------ #
    # Public port
    # ------------------------------------------------------------------ #
    def select(
        self, candidates: Sequence[CandidateContext], *, config: EngineConfig, now: datetime
    ) -> NextResult:
        """Pick the next action from ``candidates``.

        Pipeline: score every candidate -> rank desc -> compute temperature ->
        (hard-due bypass | softmax sample) -> wrap as a NextResult. ``now`` is
        accepted to satisfy the port and for symmetry with the contexts; the
        contexts already carry their own ``now`` for the terms, so we do not
        re-derive it here.
        """
        cands = list(candidates)
        if not cands:
            raise errors.NoCandidatesAvailable(
                "select() was given no candidates to choose from"
            )

        # 1. Score and rank (descending). Python's sort is stable, so ties keep
        #    their incoming order -- the ranking is fully deterministic.
        scored_pairs: list[tuple[CandidateContext, float]] = [
            (ctx, self._score(ctx, config)) for ctx in cands
        ]
        scored_pairs.sort(key=lambda pair: pair[1], reverse=True)
        ranked: tuple[ScoredCandidate, ...] = tuple(
            ScoredCandidate(concept_id=ctx.concept.id, mode=ctx.mode, score=score)
            for ctx, score in scored_pairs
        )

        # 2. Temperature. days_to_exam is shared across candidates; take the min
        #    so the result is order-independent and tracks the nearest deadline.
        horizons = [c.days_to_exam for c in cands if c.days_to_exam is not None]
        temperature = self._temperature(
            min(horizons) if horizons else None, config.base_temperature
        )

        # 3a. Hard-due bypass: take the highest-scoring overdue item with NO
        #     sampling. scored_pairs is already sorted desc, so the first
        #     hard_due we meet is the best-scoring one.
        for ctx, _score in scored_pairs:
            if ctx.hard_due:
                reason = (
                    "hard-due: chose the highest-scoring overdue concept, "
                    "bypassing the sampling lottery"
                )
                return self._result(ctx, ranked, temperature, reason)

        # 3b. Otherwise softmax-sample one of the top_k by exp(score / temp).
        k = max(1, min(self._top_k, len(scored_pairs)))
        chosen_ctx = self._sample(scored_pairs[:k], temperature)
        reason = (
            f"weighted softmax sample from the top {k} candidates "
            f"at temperature {temperature:.4f}"
        )
        return self._result(chosen_ctx, ranked, temperature, reason)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _score(self, ctx: CandidateContext, config: EngineConfig) -> float:
        """Weighted sum of the terms for one candidate.

        ``weight = config.weights.get(term.name, default_weight)``. Every term is
        ADDED except ``interleave_penalty``, which reports a positive magnitude
        and is therefore SUBTRACTED (an explicit penalty, documented here rather
        than hidden in a sign). The split keeps each term ignorant of how the
        policy composes it (Single Responsibility / Open-Closed).
        """
        total = 0.0
        for term in self._terms:
            weight = config.weights.get(term.name, self._default_weight)
            contribution = weight * term.score(ctx)
            if term.name == _INTERLEAVE_PENALTY_NAME:
                total -= contribution
            else:
                total += contribution
        return total

    def _temperature(self, days_to_exam: int | None, base_temperature: float) -> float:
        """Decay the base temperature toward the epsilon floor as the exam nears.

        ``temp = base * min(1, days_to_exam / horizon)`` so far-off (or capped)
        deadlines keep the full exploratory temperature, while an imminent exam
        (small or zero days) shrinks it toward the floor -> argmax. A ``None``
        deadline means no pressure, so the factor is 1. A non-positive horizon
        would make the ratio meaningless (or divide by zero), so we treat it as
        "no decay" and fall back to the base temperature. Past-due exams produce
        a negative ratio, which the epsilon floor turns into argmax behaviour --
        the right call once the deadline has passed.
        """
        if days_to_exam is None or self._horizon_days <= 0:
            factor = 1.0
        else:
            factor = min(1.0, days_to_exam / self._horizon_days)
        return max(_TEMP_EPSILON, base_temperature * factor)

    def _sample(
        self, top: list[tuple[CandidateContext, float]], temperature: float
    ) -> CandidateContext:
        """Softmax-sample one candidate from ``top`` using the injected RNG.

        Weights are ``exp((score - max_score) / temperature)``. Subtracting the
        max is the standard softmax-stabilisation trick: it leaves the relative
        probabilities unchanged but bounds the exponent at 0, so an enormous
        score (or a tiny temperature) can never overflow ``exp``. Because the
        max-scoring item always contributes ``exp(0) == 1``, the weight total is
        >= 1 and the inverse-CDF draw below can never divide by zero. As
        ``temperature`` approaches the floor every non-max weight collapses to
        ~0, so this degrades gracefully into a deterministic argmax.
        """
        max_score = max(score for _ctx, score in top)
        weights = [math.exp((score - max_score) / temperature) for _ctx, score in top]
        total = sum(weights)

        # Inverse-CDF sampling via a single rng.random() draw -- explicit and
        # version-stable, rather than relying on Random.choices internals.
        threshold = self._rng.random() * total
        cumulative = 0.0
        for (ctx, _score), weight in zip(top, weights):
            cumulative += weight
            if cumulative >= threshold:
                return ctx
        # Float rounding could leave cumulative a hair under threshold; fall back
        # to the last (lowest-scoring) candidate in that vanishingly rare case.
        return top[-1][0]

    @staticmethod
    def _result(
        ctx: CandidateContext,
        ranked: tuple[ScoredCandidate, ...],
        temperature: float,
        reason: str,
    ) -> NextResult:
        """Wrap a chosen candidate into a NextResult.

        ``source_refs`` are copied off the chosen concept so the tutor can ground
        whatever it teaches/asks (the anti-fabrication guarantee); ``question_id``
        is None because picking a concrete question is a later step, not this
        policy's job.
        """
        action = NextAction(
            mode=ctx.mode,
            concept_id=ctx.concept.id,
            reason=reason,
            source_refs=ctx.concept.source_refs,
            question_id=None,
        )
        return NextResult(chosen=action, candidates=ranked, temperature=temperature)

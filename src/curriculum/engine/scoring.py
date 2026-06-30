"""Selection scoring terms (the value functions behind next()).

A :class:`ScoringTerm` knows how to value ONE aspect of a candidate concept and
nothing else (Single Responsibility). Each term returns an UNWEIGHTED float in a
roughly comparable range; composing and weighting them is the SelectionPolicy's
job, not the term's -- that split is what lets a course archetype re-weight the
mix (or drop a term to zero) without touching any term's code (Open/Closed).

The five terms implement the learning-science levers the engine balances:

* urgency             -- review what is slipping below the target retention.
* difficulty_fit      -- prefer the "desirable difficulty" band (not too easy,
                         not too hard) where a retrieval is most productive.
* exploration         -- give under-visited concepts a UCB-style bonus so the
                         schedule does not starve the long tail.
* interleave_penalty  -- discourage stacking confusable items from the same
                         cluster back to back (but only once basics are solid).
* coverage            -- make sure stale and exam-relevant concepts get touched
                         before the deadline.

Everything here is a pure function of its CandidateContext: no clock reads, no
RNG, no I/O. The context already carries ``now`` (injected upstream), so the
terms stay deterministic and trivially testable. Standard library only.
"""
from __future__ import annotations

from ..domain.entities import CandidateContext
from ..domain.enums import Mastery
from ..ports.strategies import ScoringTerm

# --------------------------------------------------------------------------- #
# Tunables. Named constants (not magic numbers) so a reader -- or a future
# per-course override -- can see exactly which knob controls which behaviour.
# --------------------------------------------------------------------------- #

# A never-seen item has no retrievability signal at all; 0.5 marks it as
# moderately urgent -- worth surfacing, but not ahead of a known item that is
# actively decaying below target.
_NEVER_SEEN_URGENCY: float = 0.5

# Predicted-success sweet spot. ~0.85 is a heuristic "desirable difficulty"
# band, NOT a law of memory: retrieving when recall is high enough to usually
# succeed yet low enough to be effortful tends to strengthen memory most.
_DIFFICULTY_PEAK: float = 0.85

# When retrievability is unknown we assume a coin-flip prior for the fit term so
# a brand-new concept still gets a defined (middling) fit score.
_RETRIEVABILITY_PRIOR: float = 0.5

# Coverage staleness saturates on this time constant (days): elapsed time maps
# through days/(days+TAU) into [0,1), so coverage keeps rising with neglect but
# never runs away unbounded.
_STALENESS_TAU_DAYS: float = 30.0

# Deadline pressure saturates on this time constant (days): TAU/(TAU+d) is ~1
# when the exam is imminent and decays toward 0 when it is far off.
_DEADLINE_TAU_DAYS: float = 30.0

# Coverage blends staleness and deadline pressure additively (not as a product)
# so each driver moves the score monotonically on its own -- a freshly reviewed
# item (staleness 0) can still gain coverage as the exam nears, and vice versa.
_COVERAGE_STALENESS_WEIGHT: float = 0.5
_COVERAGE_DEADLINE_WEIGHT: float = 0.5

# Unweighted magnitude of the interleaving penalty when it fires. The policy
# decides how hard to subtract it; the term only reports "this is a same-cluster
# confusable, after basics" as a unit-magnitude flag.
_INTERLEAVE_PENALTY: float = 1.0

_SECONDS_PER_DAY: float = 86400.0


def _clamp(value: float, low: float, high: float) -> float:
    """Confine ``value`` to the inclusive range [low, high]."""
    return max(low, min(high, value))


class UrgencyTerm(ScoringTerm):
    """Value recall that is slipping below the course's target retention.

    Urgency grows as the predicted recall probability falls under the target:
    that gap is exactly the retention debt the scheduler wants to pay down. A
    never-seen item has no recall signal, so it gets a fixed moderate urgency
    rather than being treated as either fully retained or fully forgotten.
    """

    name: str = "urgency"

    def score(self, ctx: CandidateContext) -> float:
        """Return the (unweighted) retention debt, scaled to ~[0, 1].

        ``None`` retrievability -> ``_NEVER_SEEN_URGENCY`` (moderately urgent).
        Otherwise the shortfall ``target - R`` (never negative: an item already
        at/above target owes nothing) divided by ``target`` so a full miss
        (R == 0) maps to 1.0 regardless of the course's target level.
        """
        if ctx.retrievability is None:
            return _NEVER_SEEN_URGENCY
        target = ctx.profile.target_retention
        gap = max(0.0, target - ctx.retrievability)
        # Guard a degenerate (<=0) target: with no positive target there is no
        # debt to owe, so urgency is zero.
        if target <= 0.0:
            return 0.0
        return _clamp(gap / target, 0.0, 1.0)


class DifficultyFitTerm(ScoringTerm):
    """Prefer candidates near the "desirable difficulty" success band.

    The score peaks (1.0) when predicted success sits at ``_DIFFICULTY_PEAK``
    (~0.85) and falls linearly toward 0 on either side. This is a heuristic
    band, not a law: it encodes the empirical observation that retrievals which
    are likely-but-effortful build memory best, while trivially easy or
    near-impossible ones waste a slot.
    """

    name: str = "difficulty_fit"

    def score(self, ctx: CandidateContext) -> float:
        """Triangular fit centred on the peak success probability.

        Predicted success ``p`` is the retrievability, or a 0.5 prior when the
        item is unseen. ``1 - |p - peak| / peak`` is 1 at the peak and clamped
        to [0, 1] so a far-from-peak item never scores negative.
        """
        p = ctx.retrievability if ctx.retrievability is not None else _RETRIEVABILITY_PRIOR
        distance = abs(p - _DIFFICULTY_PEAK) / _DIFFICULTY_PEAK
        return _clamp(1.0 - distance, 0.0, 1.0)


class ExplorationTerm(ScoringTerm):
    """UCB-style bonus that fades as a concept accrues visits.

    ``1 / (1 + visits)`` gives an unvisited concept the maximum bonus (1.0) and
    halves it with each visit. This counteracts the rich-get-richer pull of the
    other terms so the long tail of the curriculum is not starved early on --
    the exploration half of an explore/exploit trade-off.
    """

    name: str = "exploration"

    def score(self, ctx: CandidateContext) -> float:
        """Decreasing bonus in the number of visits (never negative).

        Visits are clamped at 0 defensively so a malformed negative count can
        never blow the denominator up or invert the monotonic decay.
        """
        visits = max(0, ctx.visits)
        return 1.0 / (1.0 + visits)


class InterleavePenaltyTerm(ScoringTerm):
    """Penalise stacking same-cluster confusable items -- but only after basics.

    Interleaving is a moderator-dependent effect: it helps discrimination
    between confusable concepts, but blocking is better while a concept is still
    being acquired. So we only levy the penalty once basics are done (the
    learner has moved past NEW on this concept) AND the candidate sits in the
    same cluster the last pick came from. Otherwise there is nothing to confuse
    and the penalty is zero.

    Returns a POSITIVE magnitude; the SelectionPolicy subtracts it explicitly
    while the configured weight stays POSITIVE (the term never returns a
    negative number itself). A profile MUST keep interleave_penalty weight >= 0,
    or the explicit subtraction would turn the penalty into a reward.
    """

    name: str = "interleave_penalty"

    def score(self, ctx: CandidateContext) -> float:
        """``_INTERLEAVE_PENALTY`` when same-cluster-after-basics, else 0.0.

        Same cluster requires the candidate to actually be in a cluster and for
        that cluster to match ``extra['last_cluster']``. "After basics" requires
        an existing learner state whose mastery has advanced beyond NEW; a
        never-seen (state is None) or still-NEW concept is exempt because
        blocking, not interleaving, is what early acquisition wants.
        """
        last_cluster = ctx.extra.get("last_cluster")
        same_cluster = ctx.cluster is not None and ctx.cluster == last_cluster
        past_basics = ctx.state is not None and ctx.state.mastery is not Mastery.NEW
        return _INTERLEAVE_PENALTY if (same_cluster and past_basics) else 0.0


class CoverageTerm(ScoringTerm):
    """Push stale and exam-relevant concepts up so nothing is left untouched.

    Two independent drivers, blended additively so each is individually
    monotonic:

    * staleness  -- time since the last review, saturated through
      ``days/(days+TAU)`` into [0, 1). A never-seen concept is maximally stale
      (1.0): it has the most ground to cover.
    * deadline pressure -- ``TAU/(TAU + days_to_exam)``, ~1 when the exam is
      imminent and decaying toward 0 when it is far away (or absent).

    An additive blend (rather than a product) means a freshly reviewed concept
    can still earn coverage as the exam nears, and a long-neglected concept
    still earns it even with no deadline set -- which a product would zero out.
    """

    name: str = "coverage"

    def score(self, ctx: CandidateContext) -> float:
        """Weighted sum of staleness and deadline pressure, in [0, 1]."""
        staleness = self._staleness(ctx)
        pressure = self._deadline_pressure(ctx)
        return (
            _COVERAGE_STALENESS_WEIGHT * staleness
            + _COVERAGE_DEADLINE_WEIGHT * pressure
        )

    def _staleness(self, ctx: CandidateContext) -> float:
        """Fraction in [0, 1) growing with days since the last review.

        Never seen (no state or no last_review) is maximally stale (1.0). Elapsed
        time is floored at 0 so a clock skew (now < last_review) cannot produce a
        negative, monotonicity-breaking staleness.
        """
        if ctx.state is None or ctx.state.last_review is None:
            return 1.0
        elapsed_seconds = (ctx.now - ctx.state.last_review).total_seconds()
        days = max(0.0, elapsed_seconds / _SECONDS_PER_DAY)
        return days / (days + _STALENESS_TAU_DAYS)

    def _deadline_pressure(self, ctx: CandidateContext) -> float:
        """Fraction in [0, 1] rising as the exam nears; 0 when no exam is set.

        ``days_to_exam`` is floored at 0 so a past-due exam yields maximum
        pressure (1.0) rather than easing off.
        """
        if ctx.days_to_exam is None:
            return 0.0
        days = max(0.0, float(ctx.days_to_exam))
        return _DEADLINE_TAU_DAYS / (_DEADLINE_TAU_DAYS + days)

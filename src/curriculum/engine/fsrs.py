"""FSRS-style DSR spaced-repetition scheduler.

This is the engine's default :class:`SchedulingStrategy`. It implements the
Difficulty-Stability-Retrievability (DSR) model from the FSRS family, chosen
over the project's legacy grade-band lookup table because (per
``docs/learning-engine/learning-science-best-practices.md`` Section 5) a band
table has no stability accumulation, ignores elapsed time, conflates a one-shot
difficulty estimate with memory strength, and never adapts per item -- whereas
FSRS beats SM-2 by roughly 3x scheduling error on hundreds of millions of
reviews and rests on a peer-reviewed (KDD-2022) lineage.

Core model (two-component, Wozniak/Gorzelanczyk/Murakowski 1995):
  - Stability S: time (in days) for recall probability to fall to the FSRS
    anchor of 0.9. Larger S => the memory decays more slowly.
  - Retrievability R: probability of successful recall right now.
  - Difficulty D in [1, 10]: how resistant an item is to gaining stability.

Forgetting curve (FSRS-4.5): R(t, S) = (1 + FACTOR * t/S) ** DECAY, with
DECAY = -0.5 and FACTOR = 19/81. FACTOR is not arbitrary: it is fixed by
0.9 ** (1/DECAY) - 1 so that R(S, S) == 0.9 exactly -- i.e. one stability's worth
of elapsed time always drops recall to the 0.9 anchor.

Purity / determinism: every method is a pure function of its arguments. No
module-level randomness and no wall-clock reads; the caller injects ``now``.
Standard library only (``math``, ``datetime``).
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta

from ..domain.entities import LearnerState
from ..domain.enums import FsrsRating, Mastery
from ..ports.strategies import SchedulingStrategy

# --------------------------------------------------------------------------- #
# Forgetting-curve constants (FSRS-4.5). DECAY and FACTOR are NOT independent:
# FACTOR is derived so that R(t=S, S) == 0.9, the canonical FSRS recall anchor.
# --------------------------------------------------------------------------- #
DECAY: float = -0.5
FACTOR: float = 19.0 / 81.0  # == 0.9 ** (1 / DECAY) - 1

# Stability is floored here so it stays strictly positive: a zero/negative S
# would divide by zero in the forgetting curve and produce a degenerate (zero)
# interval. The value is tiny enough not to perturb any realistic schedule.
_S_MIN: float = 0.01

# --------------------------------------------------------------------------- #
# Published FSRS-5 default weights (19 parameters, w[0]..w[18]).
#
# IMPORTANT: these are the algorithm's *defaults*. They are meant to be tuned
# per user by fitting against that user's own review logs (gradient descent on
# log-loss); they are NOT fit to this user and should be treated as a cold-start
# prior only. Replacing W with a fitted vector is the intended customization.
#
# Index map (which weight does what in this scheduler):
#   W[0..3]  initial stability S0 for ratings Again/Hard/Good/Easy
#   W[4],W[5] initial difficulty curve (D0)
#   W[6]     difficulty change per rating step
#   W[7]     difficulty mean-reversion strength
#   W[8..10] stability-on-success: scale / saturation / retrievability gain
#   W[11..14] stability-on-lapse (post-forget) coefficients
#   W[15]    hard penalty (success), W[16] easy bonus (success)
#   W[17],W[18] short-term (same-day) stability params -- not used by this
#              long-term scheduler; kept so W matches the published 19-tuple.
# --------------------------------------------------------------------------- #
W: tuple[float, ...] = (
    0.40255,
    1.18385,
    3.173,
    15.69105,
    7.1949,
    0.5345,
    1.4604,
    0.0046,
    1.54575,
    0.1192,
    1.01925,
    1.9395,
    0.11,
    0.29605,
    2.2698,
    0.2315,
    2.9898,
    0.51655,
    0.6621,
)


def _clamp(value: float, low: float, high: float) -> float:
    """Confine ``value`` to the inclusive range [low, high]."""
    return max(low, min(high, value))


class FsrsScheduler(SchedulingStrategy):
    """A faithful FSRS-5 DSR scheduler behind the SchedulingStrategy port.

    Liskov: ``retrievability`` always returns a value in [0, 1] and ``review``
    always returns a fully-populated, schedulable :class:`LearnerState`, so any
    code written against the abstract port works unchanged with this adapter.
    """

    version: str = "fsrs-v1"

    # ----------------------------------------------------------------------- #
    # Retrievability
    # ----------------------------------------------------------------------- #
    def retrievability(self, state: LearnerState, now: datetime) -> float:
        """Probability of recall right now, in [0, 1].

        Returns 0.0 for an unseen item (no stability or never reviewed): there
        is no memory trace to recall, so the safest assumption is that it is
        gone. Elapsed time is clamped at 0 so a clock skew (now < last_review)
        cannot inflate recall above the freshly-reviewed value of 1.0.
        """
        if state.stability is None or state.last_review is None or state.stability <= 0:
            return 0.0
        elapsed_seconds = (now - state.last_review).total_seconds()
        t_days = max(0.0, elapsed_seconds / 86400.0)
        return _clamp(self._forgetting_curve(t_days, state.stability), 0.0, 1.0)

    # ----------------------------------------------------------------------- #
    # Review (the state transition)
    # ----------------------------------------------------------------------- #
    def review(
        self,
        state: LearnerState | None,
        rating: FsrsRating,
        now: datetime,
        *,
        target_retention: float,
    ) -> LearnerState:
        """Apply a graded retrieval and return the NEW learner state.

        First encounter (``state is None`` or no prior stability/last_review):
        initialise S and D directly from the published priors W -- there is no
        elapsed-time signal yet, so the success/lapse update equations do not
        apply. Subsequent reviews evolve S and D from the prior state using the
        retrievability at ``now`` (a well-timed recall at low R grows stability
        the most -- this is the spacing effect baked into the math).

        Mastery is the application layer's concern (criterion-based promotion on
        top of the schedule), so it is passed through UNCHANGED here.
        """
        is_first = (
            state is None or state.stability is None or state.last_review is None
        )

        if is_first:
            # Cold start: trust the priors. Floor S to stay strictly positive.
            new_stability = max(self._initial_stability(rating), _S_MIN)
            new_difficulty = self._initial_difficulty(rating)
            # We have no concept_id when there is no incoming state object; the
            # caller is expected to attach it. When a (stateless) state object
            # exists, preserve its identity and counters.
            concept_id = state.concept_id if state is not None else ""
            base_reps = state.reps if state is not None else 0
            base_lapses = state.lapses if state is not None else 0
            mastery = state.mastery if state is not None else Mastery.NEW
        else:
            # state is guaranteed non-None here; mypy/readers: narrow it.
            assert state is not None and state.stability is not None
            recall = self.retrievability(state, now)
            # Difficulty may be missing on a malformed state; fall back to the
            # rating's prior so the update is always well defined.
            prior_d = (
                state.difficulty
                if state.difficulty is not None
                else self._initial_difficulty(rating)
            )
            new_difficulty = self._next_difficulty(prior_d, rating)
            if rating == FsrsRating.AGAIN:
                new_stability = self._stability_on_lapse(
                    prior_d, state.stability, recall
                )
            else:
                new_stability = self._stability_on_success(
                    prior_d, state.stability, recall, rating
                )
            new_stability = max(new_stability, _S_MIN)
            concept_id = state.concept_id
            base_reps = state.reps
            base_lapses = state.lapses
            mastery = state.mastery

        # A lapse is exactly an AGAIN rating; every review counts as a rep.
        new_reps = base_reps + 1
        new_lapses = base_lapses + (1 if rating == FsrsRating.AGAIN else 0)

        # Invert the forgetting curve: pick the interval at which recall will
        # have decayed to the target retention.
        interval_days = self._interval(new_stability, target_retention)
        due_at = now + timedelta(days=interval_days)

        return LearnerState(
            concept_id=concept_id,
            stability=new_stability,
            difficulty=new_difficulty,
            last_review=now,
            due_at=due_at,
            reps=new_reps,
            lapses=new_lapses,
            mastery=mastery,  # preserved unchanged (application-layer concern)
        )

    # ----------------------------------------------------------------------- #
    # Pure helpers (the equations -- each commented with its FSRS role)
    # ----------------------------------------------------------------------- #
    def _forgetting_curve(self, t_days: float, stability: float) -> float:
        """R(t, S) = (1 + FACTOR * t/S) ** DECAY -- the FSRS-4.5 power-law decay.

        Power-law (not the older exponential 0.9 ** (t/S)) because empirically
        memory decays with a heavy tail: recall drops fast early then flattens.
        """
        return (1.0 + FACTOR * t_days / stability) ** DECAY

    def _initial_stability(self, rating: FsrsRating) -> float:
        """S0(rating) = W[rating-1]: the first-exposure stability prior.

        Better first answers earn a longer initial memory (W is increasing over
        Again<Hard<Good<Easy in the published defaults).
        """
        return W[rating - 1]

    def _initial_difficulty(self, rating: FsrsRating) -> float:
        """D0(rating) = clamp(W[4] - exp(W[5] * (rating-1)) + 1, 1, 10).

        Worse first answers start harder (higher D). Clamped to the legal [1,10]
        difficulty band that every downstream equation assumes.
        """
        raw = W[4] - math.exp(W[5] * (rating - 1)) + 1.0
        return _clamp(raw, 1.0, 10.0)

    def _next_difficulty(self, difficulty: float, rating: FsrsRating) -> float:
        """Update difficulty with linear damping + mean reversion.

        Step 1 (linear damping): delta = -W[6] * (rating - 3); a Good (3) leaves
        difficulty unchanged, harder grades raise it, easier grades lower it.
        The change is scaled by (10 - D)/9 so it shrinks as D nears the ceiling
        (FSRS-5 linear damping), preventing difficulty from slamming into 10.

        Step 2 (mean reversion): pull the result back toward the Good-rating
        anchor D0(GOOD) with strength W[7]. This is FSRS's analogue of avoiding
        SM-2 "ease hell": difficulty cannot ratchet to an extreme over a long
        history because it is continuously nudged toward a sane center.
        """
        delta = -W[6] * (rating - 3)
        damped = difficulty + delta * (10.0 - difficulty) / 9.0
        anchor = self._initial_difficulty(FsrsRating.GOOD)
        reverted = W[7] * anchor + (1.0 - W[7]) * damped
        return _clamp(reverted, 1.0, 10.0)

    def _stability_on_success(
        self,
        difficulty: float,
        stability: float,
        retrievability: float,
        rating: FsrsRating,
    ) -> float:
        """Stability after a non-Again recall (FSRS-5 SInc equation).

        S' = S * (1 + exp(W[8]) * (11 - D) * S**(-W[9])
                    * (exp(W[10] * (1 - R)) - 1) * hard_penalty * easy_bonus)

        Why each factor:
          - (11 - D): easier items gain stability faster.
          - S**(-W[9]): diminishing returns -- already-stable memories grow less.
          - (exp(W[10]*(1-R)) - 1): the spacing effect -- recalling at LOW R
            (long overdue) yields a bigger stability jump than an early review.
          - hard_penalty (W[15] < 1) damps the gain for a Hard answer;
            easy_bonus (W[16] > 1) amplifies it for an Easy answer.
        The bracket term is always positive for R < 1, so a successful recall
        strictly increases stability.
        """
        hard_penalty = W[15] if rating == FsrsRating.HARD else 1.0
        easy_bonus = W[16] if rating == FsrsRating.EASY else 1.0
        increase = (
            math.exp(W[8])
            * (11.0 - difficulty)
            * stability ** (-W[9])
            * (math.exp(W[10] * (1.0 - retrievability)) - 1.0)
            * hard_penalty
            * easy_bonus
        )
        return stability * (1.0 + increase)

    def _stability_on_lapse(
        self, difficulty: float, stability: float, retrievability: float
    ) -> float:
        """Post-lapse stability after an Again rating (FSRS-5 forget equation).

        S' = W[11] * D**(-W[12]) * ((S + 1)**W[13] - 1) * exp(W[14] * (1 - R))

        A forget collapses stability to a small value (harder items, higher D,
        collapse further); it scales sub-linearly with the prior S so a strong
        memory is not punished as severely as a weak one. This is what makes
        Again both drop the schedule and increment the lapse counter.
        """
        forget = (
            W[11]
            * difficulty ** (-W[12])
            * ((stability + 1.0) ** W[13] - 1.0)
            * math.exp(W[14] * (1.0 - retrievability))
        )
        # Mirror reference FSRS-5: a lapse must never increase stability, for
        # ALL D/S/R inputs (not just the ranges where the defaults happen to).
        return min(stability, forget)

    def _interval(self, stability: float, target_retention: float) -> float:
        """Days until recall decays to ``target_retention`` (inverts the curve).

        interval = S * (target_retention ** (1/DECAY) - 1) / FACTOR.

        Solving R(t, S) = target for t. Because FACTOR == 0.9**(1/DECAY) - 1,
        a target of 0.9 yields interval == S exactly. The interval is linear in
        S, so a more stable memory is scheduled further out -- intervals
        compound across reviews, which a flat grade-band table cannot do.
        """
        return stability * (target_retention ** (1.0 / DECAY) - 1.0) / FACTOR

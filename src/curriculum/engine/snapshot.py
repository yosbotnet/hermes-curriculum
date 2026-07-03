"""Honest game-state metrics (the numbers a learner is allowed to trust).

Where :mod:`engine.scoring` decides what to do NEXT, this module reports where
the learner STANDS right now. Each function is a pure reading of the current
domain state -- no clock reads, no RNG, no I/O; ``now`` (and any reference
instant) is always injected -- so every metric is deterministic and trivially
testable. Standard library only.

The four metrics are deliberately shaped to resist gaming and debt framing:

* ``stability_days`` -- a single "memory capital" number: the sum of every
  concept's retention stability WEIGHTED BY its exam importance. Weighting is
  the Goodhart defence: padding the corpus with trivial (low-importance)
  concepts barely moves the number, so the only way to grow it is to genuinely
  strengthen concepts that matter. A never-seen concept (``stability is None``)
  has no retention signal and contributes nothing.

* ``ripeness`` -- upcoming reviews bucketed by DUE DATE at DAY granularity into
  gain-framed names (``ready_now`` / ``ready_tomorrow`` / ``ready_this_week`` /
  ``holding``). Day granularity is intentional: retrievability decays on a day
  scale, so hour-level precision ("due in 3 hours") would manufacture fake
  urgency. Bucketing compares CALENDAR DATES (``due_at.date()`` vs
  ``now.date()``), not timedeltas, so a review due at 23:00 today is still
  ``ready_now`` at 08:00 the same day. Crucially, a review whose date has
  already passed is simply ``ready_now`` like any other: this module frames
  every item as an opportunity to harvest, never as an obligation missed,
  because gain framing invites the next rep.

* ``unlock_proximity`` -- for the concepts not yet started, how close each is to
  becoming learnable. It surfaces the "one prerequisite away" moments that make
  a curriculum feel like it is opening up.

* ``consolidation_report`` -- how much knowledge is currently HELD (intervals
  still intact) and how many reviews happened since a reference instant. Again
  a gain reading: intact intervals are knowledge retained, not a queue of
  chores.

A prerequisite counts as satisfied ("mastered") once its mastery is SOLID or
better -- the same rule as ``application.policies.is_mastered``, reimplemented
here as a named constant so the pure engine layer stays free of any dependency
on the outer application layer.
"""
from __future__ import annotations

from datetime import datetime
from typing import Mapping, Sequence

from ..domain.entities import Concept, Edge, LearnerState
from ..domain.enums import Mastery

# --------------------------------------------------------------------------- #
# Tunables and vocabularies. Named constants (not magic values) so a reader --
# or a future per-course override -- can see exactly which knob or rule governs
# which behaviour.
# --------------------------------------------------------------------------- #

# A prerequisite is "satisfied" once its mastery reaches SOLID; EXAM_READY is
# strictly stronger and also counts. This mirrors application.policies.is_mastered
# on purpose -- kept local so engine/ never imports application/.
_MASTERED: frozenset[Mastery] = frozenset({Mastery.SOLID, Mastery.EXAM_READY})

# "LEARNING or better": the learner has actually started the concept (moved past
# NEW). one_away uses this to distinguish a prereq that is nearly there (LEARNING)
# from one that has not been touched at all (NEW / no state).
_STARTED: frozenset[Mastery] = frozenset(
    {Mastery.LEARNING, Mastery.SOLID, Mastery.EXAM_READY}
)

# Ripeness day thresholds (calendar days from today). Tomorrow gets its own
# bucket; the rest of the week (2..7 days out) is "this week"; anything further
# is "holding". Seven is inclusive -- due exactly a week out is still this week.
_READY_TOMORROW_DAYS: int = 1
_READY_THIS_WEEK_DAYS: int = 7

# The ripeness bucket names, in the order the design tells the story: what can
# be gained now, next, this week, and what is safely held.
_RIPENESS_BUCKETS: tuple[str, ...] = (
    "ready_now",
    "ready_tomorrow",
    "ready_this_week",
    "holding",
)


def stability_days(
    states: Sequence[LearnerState], concepts: Mapping[str, Concept]
) -> float:
    """Importance-weighted sum of retention stability across seen concepts.

    For every state that has a retention signal (``stability is not None``) and
    whose concept is present in ``concepts``, add ``concept.importance *
    state.stability``. Never-seen states contribute nothing; a state whose
    concept_id is absent from ``concepts`` is skipped (the mapping is the source
    of importance, and there is no honest weight to apply without it). Empty
    input yields ``0.0``.

    Weighting by importance is the anti-Goodhart lever: stuffing the course with
    low-importance concepts cannot inflate this number, so the only path up is to
    strengthen concepts that actually matter for the exam.
    """
    total = 0.0
    for state in states:
        if state.stability is None:
            continue
        concept = concepts.get(state.concept_id)
        if concept is None:
            continue
        total += concept.importance * state.stability
    return total


def ripeness(
    states: Sequence[LearnerState], now: datetime
) -> Mapping[str, list[str]]:
    """Bucket scheduled reviews by due DATE into gain-framed day buckets.

    Returns a dict with exactly the four keys ``ready_now``, ``ready_tomorrow``,
    ``ready_this_week`` and ``holding``, each mapping to the concept_ids due in
    that window, sorted ascending for determinism.

    Bucketing is by CALENDAR DAY, comparing ``due_at.date()`` to ``now.date()``
    (not timedeltas): a review due at 23:00 today is ``ready_now`` at 08:00 the
    same day. The day delta decides the bucket:

    * delta <= 0  -> ``ready_now``      (due today, or any earlier date --
                                         an already-passed date is deliberately
                                         bucketed as simply ready)
    * delta == 1  -> ``ready_tomorrow``
    * delta <= 7  -> ``ready_this_week`` (within a calendar week, inclusive)
    * otherwise   -> ``holding``         (interval intact, nothing to do yet)

    States with no ``due_at`` (never scheduled) carry no ripeness signal and are
    omitted from every bucket.
    """
    buckets: dict[str, list[str]] = {name: [] for name in _RIPENESS_BUCKETS}
    today = now.date()
    for state in states:
        if state.due_at is None:
            continue
        delta_days = (state.due_at.date() - today).days
        if delta_days <= 0:
            buckets["ready_now"].append(state.concept_id)
        elif delta_days == _READY_TOMORROW_DAYS:
            buckets["ready_tomorrow"].append(state.concept_id)
        elif delta_days <= _READY_THIS_WEEK_DAYS:
            buckets["ready_this_week"].append(state.concept_id)
        else:
            buckets["holding"].append(state.concept_id)
    for name in buckets:
        buckets[name].sort()
    return buckets


def unlock_proximity(
    course_concepts: Sequence[Concept],
    states: Mapping[str, LearnerState],
    prereq_in_edges: Mapping[str, Sequence[Edge]],
) -> list[dict]:
    """How close each not-yet-started concept is to becoming learnable.

    Considers only NEVER-SEEN concepts (no state, or a state whose
    ``stability is None``): a concept already in progress has been unlocked, so
    its proximity is moot. For each such concept with AT LEAST ONE unmastered
    prerequisite, emits::

        {"concept_id": <id>, "missing": <count of unmastered prereqs>,
         "one_away": <bool>}

    where a prerequisite is "unmastered" unless its own mastery is SOLID or
    better (a prereq with no state counts as unmastered). Prerequisites are the
    ``src`` of each incoming PREREQUISITE edge in ``prereq_in_edges``, deduped by
    src so a concept is never double-counted. ``one_away`` is True only when
    EXACTLY ONE prerequisite is missing AND that prerequisite has been started
    (mastery LEARNING or better) -- i.e. the learner is genuinely on the verge,
    not staring at an untouched prereq.

    Concepts that are already fully unlocked (zero unmastered prereqs) are
    omitted. The result is sorted by ``(missing asc, concept_id asc)`` so the
    nearest, most deterministically-ordered unlocks come first.
    """
    results: list[dict] = []
    for concept in course_concepts:
        state = states.get(concept.id)
        if state is not None and state.stability is not None:
            continue  # already started -- proximity does not apply
        prereq_mastery: dict[str, Mastery] = {}
        for edge in prereq_in_edges.get(concept.id, ()):  # type: ignore[arg-type]
            src_state = states.get(edge.src)
            prereq_mastery[edge.src] = (
                src_state.mastery if src_state is not None else Mastery.NEW
            )
        unmastered = [m for m in prereq_mastery.values() if m not in _MASTERED]
        missing = len(unmastered)
        if missing == 0:
            continue  # fully unlocked -- nothing to report
        one_away = missing == 1 and unmastered[0] in _STARTED
        results.append(
            {"concept_id": concept.id, "missing": missing, "one_away": one_away}
        )
    results.sort(key=lambda row: (row["missing"], row["concept_id"]))
    return results


def consolidation_report(
    states: Sequence[LearnerState], since: datetime | None, now: datetime
) -> dict:
    """Count knowledge currently held and reviews done since a reference instant.

    Returns::

        {"holding": <count of states with due_at > now>,
         "reviewed_since": <count of states with last_review >= since>}

    ``holding`` is the number of intervals still intact -- reviews not yet due,
    i.e. knowledge the scheduler still considers held. This is a strict datetime
    comparison (``due_at > now``), NOT the calendar-day bucketing of
    :func:`ripeness`: an interval that lapses at 23:00 today is no longer holding
    the instant ``now`` passes it. States with no ``due_at`` are never holding.

    ``reviewed_since`` counts states last reviewed at or after ``since`` (an
    inclusive lower bound). When ``since`` is ``None`` there is no window to
    measure against, so ``reviewed_since`` is ``0``.
    """
    holding = sum(
        1 for state in states if state.due_at is not None and state.due_at > now
    )
    if since is None:
        reviewed_since = 0
    else:
        reviewed_since = sum(
            1
            for state in states
            if state.last_review is not None and state.last_review >= since
        )
    return {"holding": holding, "reviewed_since": reviewed_since}

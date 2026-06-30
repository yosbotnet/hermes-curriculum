"""Small, pure policy helpers used by the application service.

These encode domain decisions that are intentionally NOT in the scheduler
(which owns memory state only) nor in storage: the grade->rating mapping, the
mastery progression ladder, and a couple of id helpers. All are tunable and
documented; none reach out to I/O.
"""
from __future__ import annotations

from datetime import datetime
from typing import Protocol, Sequence

from ..domain.enums import EdgeType, FsrsRating, Mastery


class Clock(Protocol):
    """Injected time source so the service is testable with a fixed clock."""

    def now(self) -> datetime: ...


class SystemClock:
    def now(self) -> datetime:  # pragma: no cover - trivial wall-clock adapter
        from datetime import timezone

        return datetime.now(timezone.utc)


class FixedClock:
    """Deterministic clock for tests."""

    def __init__(self, instant: datetime) -> None:
        self._instant = instant

    def now(self) -> datetime:
        return self._instant


def grade_to_rating(score: int) -> FsrsRating:
    """Map a 0..6 single-question grade to a 4-point FSRS rating.

    Default mapping (tunable per course via the profile later):
      0-2 -> Again (a fail / lapse),  3 -> Hard,  4-5 -> Good,  6 -> Easy.
    Mirrors the design doc's grade-band intent on the /6 scale.
    """
    if score <= 2:
        return FsrsRating.AGAIN
    if score == 3:
        return FsrsRating.HARD
    if score <= 5:
        return FsrsRating.GOOD
    return FsrsRating.EASY


def next_mastery(score: int, recent_scores: Sequence[int]) -> Mastery:
    """Mastery ladder from grade history. `recent_scores` ends with the just-
    recorded score. Interpretation of the design's "exam-ready after two
    consecutive strong grades" on the /6 scale (tunable):
      - a fail (<=2) drops to LEARNING
      - two consecutive >=5 -> EXAM_READY
      - a >=4 -> SOLID
      - otherwise LEARNING.
    """
    if score <= 2:
        return Mastery.LEARNING
    last_two = list(recent_scores)[-2:]
    if len(last_two) == 2 and all(s >= 5 for s in last_two):
        return Mastery.EXAM_READY
    if score >= 4:
        return Mastery.SOLID
    return Mastery.LEARNING


def is_mastered(mastery: Mastery) -> bool:
    """A prerequisite counts as satisfied once SOLID or better."""
    return mastery in (Mastery.SOLID, Mastery.EXAM_READY)


def cluster_of(concept_id: str) -> str:
    """Cluster used by the interleaving penalty: the top path segment of the
    OKF concept-id (e.g. 'cyber/crypto/aes' -> 'cyber'). One level is enough to
    keep same-area picks from stacking back-to-back."""
    return concept_id.split("/", 1)[0] if "/" in concept_id else concept_id


def parse_edge_id(edge_id: str) -> tuple[str, EdgeType, str]:
    """Inverse of Edge.id == 'src::type::dst'."""
    src, type_str, dst = edge_id.split("::", 2)
    return src, EdgeType(type_str), dst

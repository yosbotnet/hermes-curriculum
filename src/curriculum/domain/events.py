"""Domain events.

Used to decouple the side-effects of grading (FSRS update, FIRe propagation,
connection-skip accounting, calibration logging) from the grade() use-case
itself: grade() emits an event, subscribers react. Keeps grade() honest to the
Single Responsibility and Open/Closed principles.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .enums import FsrsRating


@dataclass(frozen=True, slots=True)
class GradeRecorded:
    """Emitted whenever a learner answer is graded."""

    concept_id: str
    grade: int                                # 0..6
    rating: FsrsRating
    at: datetime
    question_id: str | None = None
    traversed_edges: tuple[str, ...] = ()     # edge ids the answer connected
    skipped_edges: tuple[str, ...] = ()       # relevant edge ids the answer omitted

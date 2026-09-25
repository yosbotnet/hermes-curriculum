"""The learner model (dialogue mode): what the tutor learns about the learner.

Where the concept graph models the *material*, this models the *learner*: the
abstractions they built (often in their own words), the misconceptions that were
diagnosed and fixed, the questions left open, the places the notes failed them,
and what the learning is for. The tutor writes these during a conversation and
reads them back at the start of the next one (see docs/tutor-contract.md).

Insights and fixed misconceptions are *reviewable*: they carry FSRS state, so the
engine can bring them back when they are ripe -- as the learner's own frame to be
applied to a new case, never as a definition to restate.

Like every domain object here, a note is a frozen, slotted value object updated
with ``dataclasses.replace``.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from .entities import SourceRef
from .enums import FsrsRating


class NoteKind(str, Enum):
    """The five things worth remembering about a learner."""

    INSIGHT = "insight"                          # a frame the learner built or adopted
    MISCONCEPTION_FIXED = "misconception_fixed"  # a wrong link, diagnosed and corrected
    OPEN_THREAD = "open_thread"                  # a question raised and not finished
    MATERIAL_GAP = "material_gap"                # the notes skipped a step for them
    GOAL = "goal"                                # what the learning is for


REVIEWABLE: frozenset[NoteKind] = frozenset({NoteKind.INSIGHT, NoteKind.MISCONCEPTION_FIXED})

ACTIVE = "active"
RESOLVED = "resolved"

# How a review went, in words a tutor would use, mapped onto the FSRS rating.
# A review is a transfer question on a new case, so the scale is about applying
# the frame, not reciting it.
OUTCOMES: dict[str, FsrsRating] = {
    "forgot": FsrsRating.AGAIN,       # could not apply it; needed the frame re-explained
    "struggled": FsrsRating.HARD,     # got there with hints
    "applied": FsrsRating.GOOD,       # applied it to the new case unaided
    "easy": FsrsRating.EASY,          # applied it immediately and extended it
}


@dataclass(frozen=True, slots=True)
class LearnerNote:
    """One entry of the learner model.

    ``text`` is the tutor's one-line summary; ``learner_words`` is the learner's
    own phrasing, verbatim, when there is one -- the most valuable field, since
    reviews are phrased back in it. FSRS fields are None until a reviewable note
    is scheduled (which happens the moment it is remembered).
    """

    id: str
    course: str
    kind: NoteKind
    text: str
    created_at: datetime
    topic: str = ""
    learner_words: str | None = None
    source_ref: SourceRef | None = None
    status: str = ACTIVE
    resolved_at: datetime | None = None
    stability: float | None = None
    difficulty: float | None = None
    last_review: datetime | None = None
    due_at: datetime | None = None
    reps: int = 0
    lapses: int = 0

    @property
    def reviewable(self) -> bool:
        return self.kind in REVIEWABLE and self.status == ACTIVE

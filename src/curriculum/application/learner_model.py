"""Dialogue mode: the learner-model use-cases behind the tutor's memory tools.

The tutor contract (docs/tutor-contract.md) puts the teaching in the dialogue and
the memory in the engine. This service is that memory. The tutor calls:

- ``remember``      the moment an insight lands, a misconception is fixed, a
                    question is left open, or a goal is stated;
- ``recall``        at the start of a session (and when a topic comes back);
- ``reviews``       at the start of a session: the learner's own insights that are
                    ripe, to be turned into a NEW case (never a restatement);
- ``review_result`` after the learner answers such a case;
- ``resolve``       when an open thread is answered;
- ``flag_material`` when the notes skipped a step for the learner.

Depends only on the LearnerNoteRepository port, the SchedulingStrategy port (the
same FSRS scheduler the concept graph uses) and a Clock, so it is fully testable
on the in-memory adapter with a FixedClock. Every method returns a plain,
JSON-able mapping: the MCP adapter only coerces it.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import replace
from datetime import datetime
from typing import Any, Mapping, Sequence

from ..domain.entities import LearnerState, SourceRef
from ..domain.enums import FsrsRating
from ..domain.learner import ACTIVE, OUTCOMES, RESOLVED, REVIEWABLE, LearnerNote, NoteKind
from ..ports.repositories import LearnerNoteRepository
from ..ports.strategies import SchedulingStrategy
from .policies import Clock, SystemClock

# Instructions travel with every review payload, so the tutor gets them exactly
# when it needs them (the contract's rule 6, in the payload itself).
REVIEW_HOW_TO = (
    "Do not restate this note or ask for its definition. Pose ONE new case the "
    "learner has not seen where this frame decides the answer; use their own "
    "words for the frame; withhold the answer until they try; then call "
    "review_result with forgot, struggled, applied or easy."
)

# Synonyms accepted for a review outcome (the FSRS names and 1..4).
_OUTCOME_ALIASES: dict[str, str] = {
    "again": "forgot", "1": "forgot",
    "hard": "struggled", "2": "struggled",
    "good": "applied", "3": "applied",
    "4": "easy",
}
_LIST_CAP = 8


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def note_to_dict(note: LearnerNote, now: datetime | None = None) -> dict[str, Any]:
    """Wire shape of one note. ``due_in_days`` is relative to ``now`` when given."""
    out: dict[str, Any] = {
        "id": note.id,
        "course": note.course,
        "topic": note.topic,
        "kind": note.kind.value,
        "text": note.text,
        "learner_words": note.learner_words,
        "source_ref": None if note.source_ref is None else {"file": note.source_ref.file, "line": note.source_ref.line},
        "created_at": note.created_at.isoformat(),
        "status": note.status,
    }
    if note.kind in REVIEWABLE:
        out["reviews_done"] = note.reps
        out["due_at"] = note.due_at.isoformat() if note.due_at else None
        if now is not None and note.due_at is not None:
            out["due_in_days"] = round((note.due_at - now).total_seconds() / 86400, 1)
    return out


class LearnerModelService:
    """The learner model's use-cases (dialogue mode)."""

    def __init__(
        self,
        *,
        notes: LearnerNoteRepository,
        scheduler: SchedulingStrategy,
        clock: Clock | None = None,
        target_retention: float = 0.9,
    ) -> None:
        self._notes = notes
        self._scheduler = scheduler
        self._clock = clock or SystemClock()
        self._retention = target_retention

    # ------------------------------------------------------------------ write
    def remember(
        self,
        *,
        kind: str,
        course: str,
        text: str,
        topic: str = "",
        learner_words: str | None = None,
        source_file: str | None = None,
        source_line: int | None = None,
    ) -> Mapping[str, Any]:
        """Record one entry. Idempotent: the same kind + text (case and spacing
        ignored) for the same course returns the existing active note instead of
        a duplicate, so a tutor that calls twice does no harm. Reviewable kinds
        are scheduled at once, from the FSRS cold-start prior for a GOOD first
        encounter (about three days at 0.9 retention)."""
        try:
            k = NoteKind(kind)
        except ValueError:
            raise ValueError(f"unknown kind {kind!r}; use one of {[x.value for x in NoteKind]}") from None
        if not text or not text.strip():
            raise ValueError("text must not be empty")
        if not course or not course.strip():
            raise ValueError("course must not be empty")

        for existing in self._notes.list(course, kinds=[k], status=ACTIVE):
            if _norm(existing.text) == _norm(text):
                return {**note_to_dict(existing, self._clock.now()), "created": False}

        now = self._clock.now()
        digest = hashlib.sha1(f"{course}\0{k.value}\0{_norm(text)}\0{now.isoformat()}".encode()).hexdigest()
        note = LearnerNote(
            id="ln-" + digest[:12],
            course=course.strip(),
            kind=k,
            text=text.strip(),
            created_at=now,
            topic=topic.strip(),
            learner_words=(learner_words or "").strip() or None,
            source_ref=SourceRef(source_file, source_line) if source_file else None,
        )
        if k in REVIEWABLE:
            first = self._scheduler.review(None, FsrsRating.GOOD, now, target_retention=self._retention)
            # Being learned is not a review: keep the prior's S/D/due, reps stays 0.
            note = replace(note, stability=first.stability, difficulty=first.difficulty,
                           last_review=now, due_at=first.due_at)
        self._notes.upsert(note)
        return {**note_to_dict(note, now), "created": True}

    def flag_material(
        self,
        *,
        course: str,
        source_file: str,
        what_was_unclear: str,
        source_line: int | None = None,
        topic: str = "",
    ) -> Mapping[str, Any]:
        """The notes skipped a step for the learner: remember it as a
        material_gap pointing at the place, so the notes can be fixed."""
        return self.remember(kind=NoteKind.MATERIAL_GAP.value, course=course, text=what_was_unclear,
                             topic=topic, source_file=source_file, source_line=source_line)

    def resolve(self, note_id: str, *, resolution: str | None = None) -> Mapping[str, Any]:
        """Close a note (typically an open thread that got answered). The
        resolution, if given, is appended to the note's text."""
        note = self._notes.get(note_id)
        if note is None:
            raise KeyError(f"no learner note {note_id!r}")
        text = note.text if not resolution else f"{note.text} -> {resolution.strip()}"
        done = replace(note, status=RESOLVED, resolved_at=self._clock.now(), text=text)
        self._notes.upsert(done)
        return note_to_dict(done)

    def review_result(self, note_id: str, outcome: str) -> Mapping[str, Any]:
        """Apply how a review went (forgot / struggled / applied / easy) through
        the FSRS scheduler and return the next due date."""
        note = self._notes.get(note_id)
        if note is None:
            raise KeyError(f"no learner note {note_id!r}")
        if not note.reviewable:
            raise ValueError(f"note {note_id!r} ({note.kind.value}, {note.status}) is not reviewable")
        key = _OUTCOME_ALIASES.get(str(outcome).strip().lower(), str(outcome).strip().lower())
        if key not in OUTCOMES:
            raise ValueError(f"unknown outcome {outcome!r}; use one of {list(OUTCOMES)}")
        now = self._clock.now()
        state = LearnerState(concept_id=note.id, stability=note.stability, difficulty=note.difficulty,
                             last_review=note.last_review, due_at=note.due_at,
                             reps=note.reps, lapses=note.lapses)
        new = self._scheduler.review(state, OUTCOMES[key], now, target_retention=self._retention)
        updated = replace(note, stability=new.stability, difficulty=new.difficulty,
                          last_review=now, due_at=new.due_at, reps=new.reps, lapses=new.lapses)
        self._notes.upsert(updated)
        return {
            "id": note.id,
            "outcome": key,
            "reviews_done": updated.reps,
            "next_due_at": updated.due_at.isoformat() if updated.due_at else None,
            "interval_days": round((updated.due_at - now).total_seconds() / 86400, 1) if updated.due_at else None,
        }

    # ------------------------------------------------------------------- read
    def recall(self, course: str, *, topic: str | None = None) -> Mapping[str, Any]:
        """What the tutor should know at the start of a session: goals, open
        threads, and the learner's insights and fixed misconceptions (for one
        topic when given: substring match on topic or text). Newest first."""
        now = self._clock.now()
        terms = [t for t in re.split(r"[,\s]+", (topic or "").lower()) if t]

        def on_topic(n: LearnerNote) -> bool:
            return not terms or any(t in n.topic.lower() or t in n.text.lower() for t in terms)

        def block(kind: NoteKind, filtered: bool = True) -> dict[str, Any]:
            notes = [n for n in self._notes.list(course, kinds=[kind], status=ACTIVE) if not filtered or on_topic(n)]
            notes = sorted(notes, key=lambda n: (n.created_at, n.id), reverse=True)
            return {"items": [note_to_dict(n, now) for n in notes[:_LIST_CAP]], "more": max(0, len(notes) - _LIST_CAP)}

        return {
            "course": course,
            "topic": topic,
            "goals": block(NoteKind.GOAL, filtered=False),
            "open_threads": block(NoteKind.OPEN_THREAD),
            "insights": block(NoteKind.INSIGHT),
            "misconceptions_fixed": block(NoteKind.MISCONCEPTION_FIXED),
            "material_gaps": len(self._notes.list(course, kinds=[NoteKind.MATERIAL_GAP], status=ACTIVE)),
            "ripe_reviews": len(self._notes.due(course, now)),
            "how_to_use": (
                "Open the session in at most three sentences, in the learner's words: "
                "their goal, at most one open thread, and whether reviews are ripe. "
                "Let them skip it."
            ),
        }

    def reviews(self, course: str, *, limit: int = 2) -> Mapping[str, Any]:
        """Ripe reviewable notes, most at risk first (lowest recall probability),
        each carrying the instructions for turning it into a new case."""
        now = self._clock.now()
        ripe = list(self._notes.due(course, now))

        def recall_now(n: LearnerNote) -> float:
            return self._scheduler.retrievability(
                LearnerState(concept_id=n.id, stability=n.stability, difficulty=n.difficulty,
                             last_review=n.last_review, due_at=n.due_at), now)

        ripe.sort(key=lambda n: (recall_now(n), n.id))
        items = []
        for n in ripe[: max(0, limit)]:
            d = note_to_dict(n, now)
            d["recall_probability"] = round(recall_now(n), 3)
            d["days_since_learned"] = round((now - n.created_at).total_seconds() / 86400, 1)
            items.append(d)
        return {"course": course, "reviews": items, "ripe_total": len(ripe), "how_to_use": REVIEW_HOW_TO}

    def gaps(self, course: str) -> Sequence[Mapping[str, Any]]:
        """Active material gaps, oldest first: the list of things to fix in the notes."""
        return [note_to_dict(n) for n in self._notes.list(course, kinds=[NoteKind.MATERIAL_GAP], status=ACTIVE)]

    def notes(self, course: str, *, status: str | None = ACTIVE) -> Sequence[Mapping[str, Any]]:
        """Every note of a course (for inspection), oldest first."""
        now = self._clock.now()
        return [note_to_dict(n, now) for n in self._notes.list(course, status=status)]

    def list_courses(self) -> list[str]:
        return list(self._notes.list_courses())

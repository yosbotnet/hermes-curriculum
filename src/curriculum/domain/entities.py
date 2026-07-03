"""Immutable domain entities and DTOs (the ubiquitous language of the engine).

Design notes:
- Everything is a frozen, slotted dataclass: domain objects are value objects,
  updated functionally with `dataclasses.replace`, never mutated in place. This
  removes a whole class of aliasing bugs and makes the objects safe to share.
- The OKF/Postgres split is encoded here: `Concept`/`Question` carry only
  STRUCTURE + METADATA (their authoritative home is Postgres). The prose CONTENT
  lives in OKF and is modelled separately as `ConceptContent`/`QuestionContent`,
  fetched on demand via the ContentRepository. A `Concept` therefore holds a
  `content_hash` (a pointer/staleness marker), never the body.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Mapping

from .enums import EdgeType, FsrsRating, Mastery, NextMode


@dataclass(frozen=True, slots=True)
class SourceRef:
    """A grounding pointer into the course materials (file:line). Non-empty
    source_refs are the anti-fabrication guarantee: nothing exists in the graph
    that cannot be traced to a source."""

    file: str
    line: int | None = None


# --------------------------------------------------------------------------- #
# Knowledge graph: structure + metadata (authoritative home = Postgres)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class Concept:
    """Index/metadata record for a concept. The body lives in OKF."""

    id: str                                   # OKF concept-id == bundle path without ".md"
    course: str
    title: str
    description: str = ""
    importance: float = 0.5                   # exam weight, 0..1
    source_refs: tuple[SourceRef, ...] = ()
    content_hash: str | None = None           # sha256 of the OKF content file (sync/staleness)
    status: str = "active"


@dataclass(frozen=True, slots=True)
class Edge:
    """A directed relationship between two concepts."""

    src: str
    dst: str
    type: EdgeType
    weight: float = 1.0                       # ENCOMPASSES: fraction of dst exercised, 0..1
    importance: float = 0.5                   # how exam-relevant this connection is
    rationale: str | None = None
    source_ref: SourceRef | None = None
    exposure_count: int = 0                   # times this link was relevant in context
    skip_count: int = 0                       # times the learner was near it and didn't traverse it
    last_traversed: datetime | None = None
    provenance: str = "inferred"              # spine | inferred | manual
    confidence: float = 0.6                   # trust in this link, 0..1

    @property
    def id(self) -> str:
        """Stable synthetic id used to reference an edge from a question."""
        return f"{self.src}::{self.type.value}::{self.dst}"


@dataclass(frozen=True, slots=True)
class Question:
    """Question metadata. The prompt/rubric TEXT lives in OKF (QuestionContent)."""

    id: str
    concept_id: str
    kind: str = "open"                        # mcq | open | derivation | viva ...
    difficulty: int = 1                       # 1..5
    hop_count: int = 1                        # 1 = single concept; >1 = multi-hop
    edge_id: str | None = None                # set for connection / multi-hop questions
    source_refs: tuple[SourceRef, ...] = ()
    generated_by: str | None = None
    status: str = "active"                    # active | retired (kill switch)


# --------------------------------------------------------------------------- #
# Content (authoritative home = OKF markdown bundle)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class ConceptContent:
    concept_id: str
    title: str
    body: str
    description: str = ""
    source_refs: tuple[SourceRef, ...] = ()


@dataclass(frozen=True, slots=True)
class QuestionContent:
    question_id: str
    prompt: str
    rubric: str = ""


# --------------------------------------------------------------------------- #
# Per-learner runtime state (authoritative home = Postgres; NOT knowledge -> not OKF)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class LearnerState:
    """FSRS DSR state for one (learner, concept). `None` stability == never seen."""

    concept_id: str
    stability: float | None = None
    difficulty: float | None = None
    last_review: datetime | None = None
    due_at: datetime | None = None
    reps: int = 0
    lapses: int = 0
    mastery: Mastery = Mastery.NEW


@dataclass(frozen=True, slots=True)
class ReviewEvent:
    """An append-only record of a single graded retrieval."""

    concept_id: str
    grade: int                                # 0..6 single-question scale
    fsrs_rating: FsrsRating
    at: datetime
    question_id: str | None = None
    predicted: int | None = None              # learner's self-prediction (calibration loop)
    scheduler_ver: str = "fsrs-v1"


@dataclass(frozen=True, slots=True)
class CourseProfile:
    """The frozen per-course strategy decided once at init."""

    course: str
    archetype: str
    exam_format: Mapping[str, Any] = field(default_factory=dict)
    weights: Mapping[str, float] = field(default_factory=dict)
    target_retention: float = 0.90
    exam_date: date | None = None
    confirmed_by_user: bool = False


# --------------------------------------------------------------------------- #
# Selection DTOs (what next() consumes and returns)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class CandidateContext:
    """Everything a ScoringTerm needs to score one candidate, precomputed so the
    scoring pass does no I/O."""

    concept: Concept
    mode: NextMode
    state: LearnerState | None
    retrievability: float | None              # from the scheduler; None if never seen
    now: datetime
    profile: CourseProfile
    cluster: str | None = None                # for the interleaving penalty
    visits: int = 0                           # for the exploration bonus
    days_to_exam: int | None = None           # for coverage/deadline pressure
    hard_due: bool = False                    # if True, bypass the sampling lottery
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ScoredCandidate:
    concept_id: str
    mode: NextMode
    score: float


@dataclass(frozen=True, slots=True)
class NextAction:
    mode: NextMode
    concept_id: str
    reason: str
    source_refs: tuple[SourceRef, ...] = ()
    question_id: str | None = None


@dataclass(frozen=True, slots=True)
class NextResult:
    """The engine returns the chosen action PLUS the ranked field and the
    temperature it sampled at (transparency / optional override)."""

    chosen: NextAction
    candidates: tuple[ScoredCandidate, ...]
    temperature: float


@dataclass(frozen=True, slots=True)
class EngineConfig:
    """Produced by a CourseArchetype; parameterizes the engine for one course."""

    weights: Mapping[str, float] = field(default_factory=dict)
    target_retention: float = 0.90
    enable_fire: bool = True
    enable_interleave: bool = True
    base_temperature: float = 0.6

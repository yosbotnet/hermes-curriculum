"""Strategy ports (the algorithm seams).

Every pluggable algorithm is a Strategy behind an ABC, so the engine is
Open/Closed: add an FSRS variant, a new scoring term, a different selection
policy, or a new course archetype without editing the engine that uses them.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Sequence

from ..domain.entities import (
    CandidateContext,
    CourseProfile,
    EngineConfig,
    LearnerState,
    NextResult,
)
from ..domain.enums import FsrsRating
from ..domain.events import GradeRecorded


class SchedulingStrategy(ABC):
    """Spaced-repetition timing model (FSRS today; SM-2 etc. tomorrow)."""

    version: str = "abstract"

    @abstractmethod
    def retrievability(self, state: LearnerState, now: datetime) -> float:
        """Probability of recall right now, in [0, 1]."""

    @abstractmethod
    def review(
        self,
        state: LearnerState | None,
        rating: FsrsRating,
        now: datetime,
        *,
        target_retention: float,
    ) -> LearnerState:
        """Return the updated state (new stability/difficulty/due_at/mastery)
        after a graded review. `state is None` means a first encounter."""


class ScoringTerm(ABC):
    """One additive component of the selection score. Single responsibility:
    a term knows how to value ONE aspect of a candidate (urgency, difficulty
    fit, exploration, interleaving penalty, coverage)."""

    name: str = "abstract"

    @abstractmethod
    def score(self, ctx: CandidateContext) -> float:
        """Unweighted contribution for this candidate. The SelectionPolicy
        applies the per-term weight from EngineConfig.weights[name]."""


class SelectionPolicy(ABC):
    """Turns scored candidates into a single chosen action.

    The default implementation composes weighted ScoringTerms, then samples one
    candidate via a temperature that decays toward argmax as the exam nears,
    while letting hard-due items bypass the lottery."""

    @abstractmethod
    def select(
        self, candidates: Sequence[CandidateContext], *, config: EngineConfig, now: datetime
    ) -> NextResult: ...


class CreditPropagationStrategy(ABC):
    """Maps a grade event to additional IMPLICIT rating updates on related
    concepts (Math Academy's FIRe). The Null Object (NoPropagation) returns
    nothing, so the engine path is identical whether FIRe is on or off."""

    @abstractmethod
    def propagate(self, event: GradeRecorded) -> Sequence[tuple[str, FsrsRating]]:
        """Return (concept_id, implicit_rating) updates to apply."""


class CourseArchetype(ABC):
    """A named teaching strategy template (conceptual-written, procedural,
    mcq, viva, memorization). Maps a course profile to an EngineConfig."""

    name: str = "abstract"

    @abstractmethod
    def engine_config(self, profile: CourseProfile) -> EngineConfig: ...

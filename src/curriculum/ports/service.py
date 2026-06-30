"""Application-service port: the use-case surface the MCP layer drives.

This is the boundary between "what the tutor can ask for" (next/explain/quiz/
grade/state) and "how it is computed". The MCP server is a thin transport
adapter over this interface; tests can drive it directly.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

from ..domain.entities import ConceptContent, NextResult, Question, QuestionContent


class CurriculumService(ABC):
    @abstractmethod
    def next_action(self, course: str, *, focus: str | None = None) -> NextResult:
        """Decide the single best next thing to do, plus the ranked field.

        ``focus`` (optional) scopes the candidate pool to concepts whose id or
        source token matches any of its comma/space-separated terms (e.g.
        "crypto", "cyber-03", or "m2"), so a learner can settle one topic at a
        time instead of the engine roaming the whole graph."""

    @abstractmethod
    def explain(self, concept_id: str) -> ConceptContent:
        """Return grounded prose for the tutor to teach from."""

    @abstractmethod
    def quiz(self, concept_id: str, *, difficulty: int | None = None) -> tuple[Question, QuestionContent]:
        """Pick a question for a concept and return its metadata + text."""

    @abstractmethod
    def grade(
        self,
        *,
        concept_id: str,
        score: int,
        question_id: str | None = None,
        predicted: int | None = None,
        traversed_edges: tuple[str, ...] = (),
        skipped_edges: tuple[str, ...] = (),
    ) -> Mapping[str, Any]:
        """Record a graded answer: update FSRS, run FIRe propagation, update
        connection-skip counts, log calibration. Returns the new schedule."""

    @abstractmethod
    def state(self, course: str) -> Mapping[str, Any]:
        """Burndown / progress snapshot for the course."""

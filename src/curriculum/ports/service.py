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
    def next_action(self, course: str) -> NextResult:
        """Decide the single best next thing to do, plus the ranked field."""

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

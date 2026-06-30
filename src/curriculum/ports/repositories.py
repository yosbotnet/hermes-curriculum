"""Repository ports (the storage seam).

These ABCs are the ONLY thing the application/engine know about persistence.
Concrete adapters: InMemory (tests), Postgres (structure/metadata/state), and an
OKF ContentRepository (the prose content). Per the OKF/Postgres split:
  - ConceptIndex/Edge/Question/LearnerState/ReviewLog/CourseProfile -> Postgres
  - ContentRepository -> OKF markdown bundle (source of truth for prose)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Iterable, Sequence

from ..domain.entities import (
    Concept,
    ConceptContent,
    CourseProfile,
    Edge,
    LearnerState,
    Question,
    QuestionContent,
    ReviewEvent,
)
from ..domain.enums import EdgeType


class ConceptIndexRepository(ABC):
    """Concept metadata index (NOT the body; the body lives in OKF)."""

    @abstractmethod
    def get(self, concept_id: str) -> Concept | None: ...

    @abstractmethod
    def list_by_course(self, course: str) -> Sequence[Concept]: ...

    @abstractmethod
    def upsert(self, concept: Concept) -> None: ...

    @abstractmethod
    def delete(self, concept_id: str) -> None: ...

    @abstractmethod
    def set_embedding(self, concept_id: str, vector: Sequence[float]) -> None:
        """Store the (derived) content embedding. In-memory keeps it in a dict;
        Postgres writes the pgvector column. Embeddings are a derived cache, not
        part of the OKF source of truth, which is why they live here, not in OKF."""

    @abstractmethod
    def nearest(
        self, vector: Sequence[float], *, course: str, k: int = 5
    ) -> Sequence[tuple[str, float]]:
        """Return up to k (concept_id, similarity) nearest to the given vector
        (for semantic dedup/search). Similarity convention: higher == closer."""

    @abstractmethod
    def nearest_to(
        self, concept_id: str, *, course: str, k: int = 10
    ) -> Sequence[tuple[str, float]]:
        """Concepts nearest to an EXISTING concept's stored embedding (excluding
        the concept itself), highest similarity first; empty if it has no stored
        embedding. Lets embedding-guided edge linking find candidate connections
        deterministically, without an LLM guessing across the whole graph."""


class EdgeRepository(ABC):
    """The knowledge graph's edges (Postgres is authoritative)."""

    @abstractmethod
    def upsert(self, edge: Edge) -> None: ...

    @abstractmethod
    def get(self, src: str, dst: str, type: EdgeType) -> Edge | None: ...

    @abstractmethod
    def out_edges(self, src: str, type: EdgeType | None = None) -> Sequence[Edge]: ...

    @abstractmethod
    def in_edges(self, dst: str, type: EdgeType | None = None) -> Sequence[Edge]: ...

    @abstractmethod
    def list_by_course(self, course: str) -> Sequence[Edge]: ...

    @abstractmethod
    def record_exposure(
        self, src: str, dst: str, type: EdgeType, *, skipped: bool, at: datetime
    ) -> None:
        """Increment exposure_count, and skip_count when `skipped` is True."""


class QuestionRepository(ABC):
    @abstractmethod
    def get(self, question_id: str) -> Question | None: ...

    @abstractmethod
    def upsert(self, question: Question) -> None: ...

    @abstractmethod
    def by_concept(
        self, concept_id: str, *, difficulty: int | None = None, hop_count: int | None = None
    ) -> Sequence[Question]: ...

    @abstractmethod
    def by_edge(self, edge_id: str) -> Sequence[Question]: ...


class LearnerStateRepository(ABC):
    @abstractmethod
    def get(self, concept_id: str) -> LearnerState | None: ...

    @abstractmethod
    def upsert(self, state: LearnerState) -> None: ...

    @abstractmethod
    def due(self, course: str, before: datetime) -> Sequence[LearnerState]: ...

    @abstractmethod
    def all_for_course(self, course: str) -> Sequence[LearnerState]: ...


class ReviewLogRepository(ABC):
    @abstractmethod
    def append(self, event: ReviewEvent) -> None: ...

    @abstractmethod
    def by_concept(self, concept_id: str) -> Sequence[ReviewEvent]: ...


class CourseProfileRepository(ABC):
    @abstractmethod
    def get(self, course: str) -> CourseProfile | None: ...

    @abstractmethod
    def upsert(self, profile: CourseProfile) -> None: ...


class ContentRepository(ABC):
    """OKF-backed prose store. The authoritative home for concept/question text.

    Implementations read/write a directory tree of markdown files with YAML
    frontmatter (OKF v0.1). `put_*` return the content hash (sha256) so the
    sync service can detect staleness against the Postgres index.
    """

    @abstractmethod
    def get_concept_content(self, concept_id: str) -> ConceptContent | None: ...

    @abstractmethod
    def put_concept_content(self, content: ConceptContent) -> str: ...

    @abstractmethod
    def get_question_content(self, question_id: str) -> QuestionContent | None: ...

    @abstractmethod
    def put_question_content(self, content: QuestionContent) -> str: ...

    @abstractmethod
    def iter_concepts(self) -> Iterable[tuple[str, str]]:
        """Yield (concept_id, content_hash) for every concept doc in the bundle."""

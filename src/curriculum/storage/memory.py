"""In-memory, dict-backed adapters for every repository port.

These are the reference adapters used by the test-suite and offline runs: they
have no external dependencies (pure stdlib) and are the executable specification
of what the Postgres/OKF adapters must reproduce. Two design choices are worth
calling out:

- Domain entities are frozen, slotted value objects, so storing a reference is
  safe (no aliasing/mutation bugs); we never defensively deep-copy them. The one
  exception is embedding vectors, which arrive as mutable sequences and are
  snapshotted into tuples so a caller cannot mutate stored state after the fact.
- Neither ``Edge`` nor ``LearnerState`` carries a ``course`` field (course is a
  property of a *concept*, not of an edge or a per-learner state). Both adapters
  therefore take a ``ConceptIndexRepository`` and resolve a concept's course
  through it, keeping the single source of truth in the concept index rather than
  duplicating ``course`` onto every row.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, replace
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
from ..ports.repositories import (
    ConceptIndexRepository,
    ContentRepository,
    CourseProfileRepository,
    EdgeRepository,
    LearnerStateRepository,
    QuestionRepository,
    ReviewLogRepository,
)

__all__ = [
    "InMemoryConceptIndexRepository",
    "InMemoryEdgeRepository",
    "InMemoryQuestionRepository",
    "InMemoryLearnerStateRepository",
    "InMemoryReviewLogRepository",
    "InMemoryCourseProfileRepository",
    "InMemoryContentRepository",
]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity in [-1, 1]; 0.0 for a zero-magnitude vector.

    A zero vector has no direction, so similarity is undefined; we return 0.0
    rather than dividing by zero. ``zip`` truncates to the shorter length, which
    keeps the function total even if a stray vector has the wrong dimension.
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _content_hash(content: object) -> str:
    """Stable sha256 hex of a content dataclass.

    The payload is canonicalised with sorted keys and a fixed separator so the
    digest depends only on the *values*, never on field declaration order or
    whitespace. ``asdict`` recurses into nested ``SourceRef`` tuples, so any
    change to the body, title, refs, prompt or rubric flips the hash; identical
    content always yields the identical digest (the sync layer relies on this to
    detect staleness against the Postgres index)."""
    payload = json.dumps(
        asdict(content), sort_keys=True, ensure_ascii=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class InMemoryConceptIndexRepository(ConceptIndexRepository):
    """Dict-backed concept index plus an embedding cache for semantic search."""

    def __init__(self) -> None:
        self._concepts: dict[str, Concept] = {}
        self._embeddings: dict[str, tuple[float, ...]] = {}

    def get(self, concept_id: str) -> Concept | None:
        return self._concepts.get(concept_id)

    def list_by_course(self, course: str) -> Sequence[Concept]:
        # Sorted by id for deterministic iteration order across runs.
        return [c for _, c in sorted(self._concepts.items()) if c.course == course]

    def upsert(self, concept: Concept) -> None:
        self._concepts[concept.id] = concept

    def delete(self, concept_id: str) -> None:
        # Drop the derived embedding too: it is a cache keyed on this concept.
        self._concepts.pop(concept_id, None)
        self._embeddings.pop(concept_id, None)

    def set_embedding(self, concept_id: str, vector: Sequence[float]) -> None:
        # Snapshot into an immutable tuple so later caller-side mutation of the
        # passed list cannot corrupt stored state.
        self._embeddings[concept_id] = tuple(float(x) for x in vector)

    def nearest(
        self, vector: Sequence[float], *, course: str, k: int = 5
    ) -> Sequence[tuple[str, float]]:
        if k <= 0:
            return []
        query = tuple(float(x) for x in vector)
        scored: list[tuple[str, float]] = []
        for concept_id, stored in self._embeddings.items():
            concept = self._concepts.get(concept_id)
            # Filter by course; skip embeddings whose concept is unknown here
            # (cannot be attributed to a course, so cannot match the filter).
            if concept is None or concept.course != course:
                continue
            scored.append((concept_id, _cosine(query, stored)))
        # Highest similarity first; ties broken by id so results are stable.
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[:k]

    def nearest_to(
        self, concept_id: str, *, course: str, k: int = 10
    ) -> Sequence[tuple[str, float]]:
        vec = self._embeddings.get(concept_id)
        if vec is None or k <= 0:
            return []
        # Ask for one extra, then drop the concept itself.
        return [
            (cid, sim)
            for cid, sim in self.nearest(vec, course=course, k=k + 1)
            if cid != concept_id
        ][:k]


class InMemoryEdgeRepository(EdgeRepository):
    """Dict-backed edge store keyed by the synthetic ``Edge.id``.

    Takes a ``ConceptIndexRepository`` because edges have no ``course`` of their
    own: ``list_by_course`` resolves an edge to the course of its *source*
    concept (the graph is intra-course and directed, so src is authoritative)."""

    def __init__(self, concepts: ConceptIndexRepository) -> None:
        self._concepts = concepts
        self._edges: dict[str, Edge] = {}

    def upsert(self, edge: Edge) -> None:
        self._edges[edge.id] = edge

    def get(self, src: str, dst: str, type: EdgeType) -> Edge | None:
        return self._edges.get(f"{src}::{type.value}::{dst}")

    def out_edges(self, src: str, type: EdgeType | None = None) -> Sequence[Edge]:
        out = [
            e
            for e in self._edges.values()
            if e.src == src and (type is None or e.type == type)
        ]
        out.sort(key=lambda e: e.id)
        return out

    def in_edges(self, dst: str, type: EdgeType | None = None) -> Sequence[Edge]:
        incoming = [
            e
            for e in self._edges.values()
            if e.dst == dst and (type is None or e.type == type)
        ]
        incoming.sort(key=lambda e: e.id)
        return incoming

    def list_by_course(self, course: str) -> Sequence[Edge]:
        result: list[Edge] = []
        for edge in self._edges.values():
            src_concept = self._concepts.get(edge.src)
            if src_concept is not None and src_concept.course == course:
                result.append(edge)
        result.sort(key=lambda e: e.id)
        return result

    def record_exposure(
        self, src: str, dst: str, type: EdgeType, *, skipped: bool, at: datetime
    ) -> None:
        """Account a traversal opportunity on an edge.

        ``exposure_count`` rises every time the link was relevant in context;
        ``skip_count`` only when the learner was near it and did not traverse it;
        ``last_traversed`` advances only on an actual (non-skipped) traversal.
        The update is a functional ``replace`` of the stored edge. If the edge
        is not yet stored we start from a default ``Edge`` so this accounting
        operation is total and never depends on ingestion ordering."""
        current = self.get(src, dst, type) or Edge(src=src, dst=dst, type=type)
        updated = replace(
            current,
            exposure_count=current.exposure_count + 1,
            skip_count=current.skip_count + (1 if skipped else 0),
            last_traversed=current.last_traversed if skipped else at,
        )
        self._edges[updated.id] = updated


class InMemoryQuestionRepository(QuestionRepository):
    """Dict-backed question-metadata store with concept/edge lookups."""

    def __init__(self) -> None:
        self._questions: dict[str, Question] = {}

    def get(self, question_id: str) -> Question | None:
        return self._questions.get(question_id)

    def upsert(self, question: Question) -> None:
        self._questions[question.id] = question

    def by_concept(
        self,
        concept_id: str,
        *,
        difficulty: int | None = None,
        hop_count: int | None = None,
    ) -> Sequence[Question]:
        # Optional filters are exact-match and independent (AND semantics).
        out = [
            q
            for q in self._questions.values()
            if q.concept_id == concept_id
            and (difficulty is None or q.difficulty == difficulty)
            and (hop_count is None or q.hop_count == hop_count)
        ]
        out.sort(key=lambda q: q.id)
        return out

    def by_edge(self, edge_id: str) -> Sequence[Question]:
        out = [q for q in self._questions.values() if q.edge_id == edge_id]
        out.sort(key=lambda q: q.id)
        return out


class InMemoryLearnerStateRepository(LearnerStateRepository):
    """Dict-backed FSRS state store keyed by ``concept_id``.

    ``LearnerState`` has no ``course`` field, so this adapter resolves a state's
    course through the injected ``ConceptIndexRepository`` for ``due`` and
    ``all_for_course``. States whose concept is absent from the index are
    excluded (their course cannot be determined)."""

    def __init__(self, concepts: ConceptIndexRepository) -> None:
        self._concepts = concepts
        self._states: dict[str, LearnerState] = {}

    def get(self, concept_id: str) -> LearnerState | None:
        return self._states.get(concept_id)

    def upsert(self, state: LearnerState) -> None:
        self._states[state.concept_id] = state

    def _in_course(self, concept_id: str, course: str) -> bool:
        concept = self._concepts.get(concept_id)
        return concept is not None and concept.course == course

    def due(self, course: str, before: datetime) -> Sequence[LearnerState]:
        # Only scheduled states (due_at set) that have come due by `before`.
        out = [
            s
            for s in self._states.values()
            if s.due_at is not None
            and s.due_at <= before
            and self._in_course(s.concept_id, course)
        ]
        out.sort(key=lambda s: (s.due_at, s.concept_id))
        return out

    def all_for_course(self, course: str) -> Sequence[LearnerState]:
        out = [
            s for s in self._states.values() if self._in_course(s.concept_id, course)
        ]
        out.sort(key=lambda s: s.concept_id)
        return out


class InMemoryReviewLogRepository(ReviewLogRepository):
    """Append-only review log, grouped lazily by concept on read."""

    def __init__(self) -> None:
        self._events: list[ReviewEvent] = []

    def append(self, event: ReviewEvent) -> None:
        self._events.append(event)

    def by_concept(self, concept_id: str) -> Sequence[ReviewEvent]:
        # Preserve append order: the log is a time series for one concept.
        return [e for e in self._events if e.concept_id == concept_id]


class InMemoryCourseProfileRepository(CourseProfileRepository):
    """Dict-backed store for the one frozen profile per course."""

    def __init__(self) -> None:
        self._profiles: dict[str, CourseProfile] = {}

    def get(self, course: str) -> CourseProfile | None:
        return self._profiles.get(course)

    def upsert(self, profile: CourseProfile) -> None:
        self._profiles[profile.course] = profile


class InMemoryContentRepository(ContentRepository):
    """In-memory stand-in for the OKF prose bundle.

    ``put_*`` compute and cache the content sha256 (the same digest the OKF/file
    adapter would derive from the on-disk markdown) and return it so the sync
    service can compare against the Postgres index. ``iter_concepts`` replays the
    stored (concept_id, hash) pairs the way a directory walk would."""

    def __init__(self) -> None:
        self._concept_content: dict[str, ConceptContent] = {}
        self._concept_hash: dict[str, str] = {}
        self._question_content: dict[str, QuestionContent] = {}

    def get_concept_content(self, concept_id: str) -> ConceptContent | None:
        return self._concept_content.get(concept_id)

    def put_concept_content(self, content: ConceptContent) -> str:
        digest = _content_hash(content)
        self._concept_content[content.concept_id] = content
        self._concept_hash[content.concept_id] = digest
        return digest

    def get_question_content(self, question_id: str) -> QuestionContent | None:
        return self._question_content.get(question_id)

    def put_question_content(self, content: QuestionContent) -> str:
        # Questions need no staleness index of their own here, but the hash is
        # still the adapter's return contract (parity with concept content).
        self._question_content[content.question_id] = content
        return _content_hash(content)

    def iter_concepts(self) -> Iterable[tuple[str, str]]:
        # Sorted for deterministic enumeration, mirroring a stable bundle walk.
        for concept_id in sorted(self._concept_hash):
            yield concept_id, self._concept_hash[concept_id]

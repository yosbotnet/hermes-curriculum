"""Composition roots: the ONE place concrete adapters are wired to the service.

Keeping construction here (and nowhere else) means the rest of the code depends
only on ports. `build_in_memory` gives a fully-wired offline stack for tests and
dry runs; `build_service` is the production wiring (Postgres + OKF) the MCP
server boots.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from ..archetypes.registry import engine_config_for
from ..config import Settings, load
from ..engine.fire import FirePropagation
from ..engine.fsrs import FsrsScheduler
from ..engine.scoring import (
    CoverageTerm,
    DifficultyFitTerm,
    ExplorationTerm,
    InterleavePenaltyTerm,
    UrgencyTerm,
)
from ..engine.selection import WeightedSamplingPolicy
from ..ports.repositories import (
    ConceptIndexRepository,
    ContentRepository,
    CourseProfileRepository,
    EdgeRepository,
    LearnerStateRepository,
    QuestionRepository,
    ReviewLogRepository,
    TelemetryRepository,
)
from ..ports.service import CurriculumService
from ..ports.strategies import ScoringTerm
from ..storage.memory import (
    InMemoryConceptIndexRepository,
    InMemoryContentRepository,
    InMemoryCourseProfileRepository,
    InMemoryEdgeRepository,
    InMemoryLearnerStateRepository,
    InMemoryQuestionRepository,
    InMemoryReviewLogRepository,
    InMemoryTelemetryRepository,
)
from .policies import Clock, SystemClock
from .service import CurriculumApplicationService


def default_terms() -> list[ScoringTerm]:
    """The five scoring terms in canonical order. Add a term here (plus a weight
    in the archetypes) to extend the selection score -- Open/Closed."""
    return [
        UrgencyTerm(),
        DifficultyFitTerm(),
        ExplorationTerm(),
        InterleavePenaltyTerm(),
        CoverageTerm(),
    ]


@dataclass(slots=True)
class InMemoryStack:
    """Handles to the wired service and its repos, so tests/dry-runs can seed."""

    service: CurriculumService
    concepts: ConceptIndexRepository
    edges: EdgeRepository
    questions: QuestionRepository
    states: LearnerStateRepository
    reviews: ReviewLogRepository
    profiles: CourseProfileRepository
    content: ContentRepository
    telemetry: TelemetryRepository


def build_in_memory(
    *,
    rng: random.Random | None = None,
    clock: Clock | None = None,
    telemetry: TelemetryRepository | None = None,
) -> InMemoryStack:
    """A fully-wired stack on in-memory repositories. No Postgres, no inference."""
    concepts = InMemoryConceptIndexRepository()
    edges = InMemoryEdgeRepository(concepts)
    questions = InMemoryQuestionRepository()
    states = InMemoryLearnerStateRepository(concepts)
    reviews = InMemoryReviewLogRepository()
    profiles = InMemoryCourseProfileRepository()
    content = InMemoryContentRepository()
    telemetry = telemetry or InMemoryTelemetryRepository()
    service = CurriculumApplicationService(
        concepts=concepts,
        edges=edges,
        questions=questions,
        states=states,
        reviews=reviews,
        profiles=profiles,
        content=content,
        scheduler=FsrsScheduler(),
        selection=WeightedSamplingPolicy(default_terms(), rng=rng or random.Random(0)),
        propagation=FirePropagation(edges),
        resolve_config=engine_config_for,
        telemetry=telemetry,
        clock=clock or SystemClock(),
    )
    return InMemoryStack(
        service, concepts, edges, questions, states, reviews, profiles, content, telemetry
    )


def build_service(settings: Settings | None = None) -> CurriculumService:
    """Production wiring: Postgres-backed structure/state + an OKF content bundle.
    Imported lazily so the in-memory path needs neither psycopg nor a bundle."""
    settings = settings or load()
    from ..storage.okf_content import FileContentRepository
    from ..storage.postgres import PostgresRepositories, connect

    repos = PostgresRepositories(connect(settings.database_url))
    content = FileContentRepository(Path(settings.okf_bundle_path))
    return CurriculumApplicationService(
        concepts=repos.concepts,
        edges=repos.edges,
        questions=repos.questions,
        states=repos.learner_state,
        reviews=repos.review_log,
        profiles=repos.profiles,
        content=content,
        scheduler=FsrsScheduler(),
        selection=WeightedSamplingPolicy(default_terms(), rng=random.Random()),
        propagation=FirePropagation(repos.edges),
        resolve_config=engine_config_for,
        telemetry=repos.telemetry,
        clock=SystemClock(),
    )

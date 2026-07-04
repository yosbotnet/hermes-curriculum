"""Postgres + pgvector adapters for every Postgres-owned repository port.

This is the production counterpart to the in-memory reference adapters: the
in-memory store (``storage.memory``) is the executable specification, and these
adapters must be Liskov-substitutable for it. Postgres owns graph structure,
metadata, derived embeddings, and per-learner state (the OKF bundle owns prose,
served by a separate ``ContentRepository`` -- hence there is deliberately no
Postgres ``ContentRepository`` here).

Driver guard
------------
``psycopg`` (v3) and ``pgvector`` are optional, heavyweight dependencies. The
module must still import on a machine that has neither (so the rest of the
package, its tests, and offline runs are unaffected). We therefore import the
driver inside a ``try`` and fall back to ``None``; the concrete adapters only
run their SQL when a real connection is handed in, which is impossible without
the driver, so the guarded ``None`` sentinels are never dereferenced in anger.

Two schema facts shape the mapping (see schema/001_init.sql):
- ``source_refs`` is ``jsonb``; we round-trip it to ``tuple[SourceRef, ...]``.
- Neither ``edge`` nor ``learner_state`` carries a ``course`` column (course is
  a property of a *concept*), so course-scoped queries JOIN ``concept`` on the
  relevant concept id -- the single source of truth stays in ``concept``.
"""
from __future__ import annotations

from datetime import datetime
from typing import Sequence

from ..domain.entities import (
    Concept,
    CourseProfile,
    Edge,
    LearnerState,
    Question,
    ReviewEvent,
    SourceRef,
)
from ..domain.enums import EdgeType, FsrsRating, Mastery
from ..domain.telemetry import EngagementEvent
from ..ports.repositories import (
    ConceptIndexRepository,
    CourseProfileRepository,
    EdgeRepository,
    LearnerStateRepository,
    QuestionRepository,
    ReviewLogRepository,
    TelemetryRepository,
)

# --------------------------------------------------------------------------- #
# Optional-driver guard: the module imports cleanly even with no driver/DB.
# pgvector lives behind psycopg, so if either import fails we disable the whole
# adapter by setting every driver symbol to None.
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - exercised only where the driver is installed
    import psycopg
    from psycopg.types.json import Jsonb
    from pgvector.psycopg import register_vector
except ImportError:  # pragma: no cover - the import-without-driver path
    psycopg = None
    Jsonb = None
    register_vector = None

__all__ = [
    "PostgresRepositories",
    "PostgresConceptIndexRepository",
    "PostgresEdgeRepository",
    "PostgresQuestionRepository",
    "PostgresLearnerStateRepository",
    "PostgresReviewLogRepository",
    "PostgresTelemetryRepository",
    "PostgresCourseProfileRepository",
    "connect",
]

# Explicit column orders: SELECTs below depend on positional unpacking, so the
# order is part of the contract between each query and its row->entity mapper.
_CONCEPT_COLS = (
    "id, course, title, description, importance, source_refs, content_hash, status"
)
_EDGE_COLS = (
    "src, dst, type, weight, importance, rationale, source_ref, "
    "exposure_count, skip_count, last_traversed, provenance, confidence"
)
_QUESTION_COLS = (
    "id, concept_id, kind, difficulty, hop_count, edge_id, source_refs, "
    "generated_by, status"
)
_STATE_COLS = (
    "concept_id, stability, difficulty, last_review, due_at, reps, lapses, mastery"
)
_EVENT_COLS = (
    "concept_id, grade, fsrs_rating, at, question_id, predicted, scheduler_ver"
)
_PROFILE_COLS = (
    "course, archetype, exam_format, weights, target_retention, "
    "exam_date, confirmed_by_user"
)
_ENGAGEMENT_COLS = "kind, course, at, payload"


# --------------------------------------------------------------------------- #
# Pure mapping helpers (no I/O, no driver) -- unit-testable without a database.
# --------------------------------------------------------------------------- #
def _refs_to_json(refs: Sequence[SourceRef]) -> list[dict]:
    """Serialise a SourceRef tuple to the jsonb shape stored in the column."""
    return [{"file": r.file, "line": r.line} for r in refs]


def _refs_from_json(data: object) -> tuple[SourceRef, ...]:
    """Rebuild a SourceRef tuple from a parsed jsonb list.

    Tolerant of NULL/empty so the mapping is total: a missing or empty column
    deserialises to the empty tuple, matching the entity default."""
    if not data:
        return ()
    return tuple(
        SourceRef(file=item["file"], line=item.get("line")) for item in data
    )


def _ref_to_json(ref: SourceRef | None) -> dict | None:
    """Serialise an optional single SourceRef (edge.source_ref) for jsonb."""
    if ref is None:
        return None
    return {"file": ref.file, "line": ref.line}


def _ref_from_json(data: object) -> SourceRef | None:
    """Rebuild an optional single SourceRef from a parsed jsonb object."""
    if not data:
        return None
    return SourceRef(file=data["file"], line=data.get("line"))


def _vector_literal(vector: Sequence[float]) -> str:
    """Format a vector as pgvector's text literal ('[a,b,c]').

    We build the literal ourselves and pair it with an explicit ``::vector``
    cast at the call site, rather than relying on the pgvector adapter to dump a
    Python ``list``. That keeps writes/queries correct regardless of whether the
    caller passes a list, tuple, or any float sequence, and avoids a hard
    dependency on numpy for the dump path."""
    return "[" + ",".join(repr(float(x)) for x in vector) + "]"


def _similarity_from_distance(distance: float) -> float:
    """Map an L2 distance (pgvector ``<->``) to a similarity where higher==closer.

    The repository contract returns a *similarity* (higher == nearer), but
    ``<->`` yields a *distance* (lower == nearer). ``1 / (1 + d)`` is monotone
    decreasing in ``d``, maps an exact hit (d=0) to 1.0, and stays in (0, 1],
    giving a stable, bounded score without needing a max-distance normaliser."""
    return 1.0 / (1.0 + float(distance))


def _row_to_concept(row: tuple) -> Concept:
    return Concept(
        id=row[0],
        course=row[1],
        title=row[2],
        description=row[3],
        importance=row[4],
        source_refs=_refs_from_json(row[5]),
        content_hash=row[6],
        status=row[7],
    )


def _row_to_edge(row: tuple) -> Edge:
    return Edge(
        src=row[0],
        dst=row[1],
        type=EdgeType(row[2]),
        weight=row[3],
        importance=row[4],
        rationale=row[5],
        source_ref=_ref_from_json(row[6]),
        exposure_count=row[7],
        skip_count=row[8],
        last_traversed=row[9],
        provenance=row[10],
        confidence=row[11],
    )


def _row_to_question(row: tuple) -> Question:
    return Question(
        id=row[0],
        concept_id=row[1],
        kind=row[2],
        difficulty=row[3],
        hop_count=row[4],
        edge_id=row[5],
        source_refs=_refs_from_json(row[6]),
        generated_by=row[7],
        status=row[8],
    )


def _row_to_state(row: tuple) -> LearnerState:
    return LearnerState(
        concept_id=row[0],
        stability=row[1],
        difficulty=row[2],
        last_review=row[3],
        due_at=row[4],
        reps=row[5],
        lapses=row[6],
        mastery=Mastery(row[7]),
    )


def _row_to_event(row: tuple) -> ReviewEvent:
    return ReviewEvent(
        concept_id=row[0],
        grade=row[1],
        fsrs_rating=FsrsRating(row[2]),
        at=row[3],
        question_id=row[4],
        predicted=row[5],
        scheduler_ver=row[6],
    )


def _row_to_engagement(row: tuple) -> EngagementEvent:
    # payload is jsonb; psycopg parses it to a dict. Coalesce a NULL/absent
    # payload to an empty dict to match the entity's default.
    return EngagementEvent(
        kind=row[0],
        course=row[1],
        at=row[2],
        payload=row[3] or {},
    )


def _row_to_profile(row: tuple) -> CourseProfile:
    return CourseProfile(
        course=row[0],
        archetype=row[1],
        exam_format=row[2] or {},
        weights=row[3] or {},
        target_retention=row[4],
        exam_date=row[5],
        confirmed_by_user=row[6],
    )


class _PgRepo:
    """Shared base: every adapter holds the connection and runs SQL through it.

    The repositories are intentionally transaction-agnostic: they neither open
    nor commit transactions. Commit/rollback is the caller's unit-of-work
    decision, which keeps a multi-repository operation atomic over one
    connection (the hexagonal "transaction script" lives above this seam)."""

    def __init__(self, conn) -> None:
        self._conn = conn


class PostgresConceptIndexRepository(_PgRepo, ConceptIndexRepository):
    """Concept metadata index backed by the ``concept`` table + pgvector column."""

    def get(self, concept_id: str) -> Concept | None:
        cur = self._conn.execute(
            f"SELECT {_CONCEPT_COLS} FROM concept WHERE id = %s", (concept_id,)
        )
        row = cur.fetchone()
        return _row_to_concept(row) if row is not None else None

    def list_by_course(self, course: str) -> Sequence[Concept]:
        # ORDER BY id mirrors the in-memory adapter's deterministic ordering.
        cur = self._conn.execute(
            f"SELECT {_CONCEPT_COLS} FROM concept WHERE course = %s ORDER BY id",
            (course,),
        )
        return [_row_to_concept(r) for r in cur.fetchall()]

    def list_courses(self) -> Sequence[str]:
        cur = self._conn.execute(
            "SELECT DISTINCT course FROM concept ORDER BY course"
        )
        return [row[0] for row in cur.fetchall()]

    def upsert(self, concept: Concept) -> None:
        # The update list deliberately omits ``embedding``: it is a derived cache
        # refreshed by set_embedding, and an upsert of metadata must not wipe it.
        self._conn.execute(
            """
            INSERT INTO concept
                (id, course, title, description, importance, source_refs,
                 content_hash, status, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, now())
            ON CONFLICT (id) DO UPDATE SET
                course = EXCLUDED.course,
                title = EXCLUDED.title,
                description = EXCLUDED.description,
                importance = EXCLUDED.importance,
                source_refs = EXCLUDED.source_refs,
                content_hash = EXCLUDED.content_hash,
                status = EXCLUDED.status,
                updated_at = now()
            """,
            (
                concept.id,
                concept.course,
                concept.title,
                concept.description,
                concept.importance,
                Jsonb(_refs_to_json(concept.source_refs)),
                concept.content_hash,
                concept.status,
            ),
        )

    def delete(self, concept_id: str) -> None:
        # FK ON DELETE CASCADE drops dependent edges/questions/state/log rows;
        # the embedding lives on the concept row, so it goes too.
        self._conn.execute("DELETE FROM concept WHERE id = %s", (concept_id,))

    def set_embedding(self, concept_id: str, vector: Sequence[float]) -> None:
        self._conn.execute(
            "UPDATE concept SET embedding = %s::vector, updated_at = now() "
            "WHERE id = %s",
            (_vector_literal(vector), concept_id),
        )

    def nearest(
        self, vector: Sequence[float], *, course: str, k: int = 5
    ) -> Sequence[tuple[str, float]]:
        # Postgres rejects a negative LIMIT; an empty request is an empty result.
        if k <= 0:
            return []
        literal = _vector_literal(vector)
        cur = self._conn.execute(
            """
            SELECT id, embedding <=> %s::vector AS distance
            FROM concept
            WHERE course = %s AND embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector, id
            LIMIT %s
            """,
            (literal, course, literal, k),
        )
        # Cosine distance in [0, 2]; similarity = 1 - distance matches the
        # in-memory reference's cosine similarity (higher == closer). The ", id"
        # tie-break makes equidistant rows deterministic across both adapters.
        return [(row[0], 1.0 - row[1]) for row in cur.fetchall()]

    def nearest_to(
        self, concept_id: str, *, course: str, k: int = 10
    ) -> Sequence[tuple[str, float]]:
        if k <= 0:
            return []
        cur = self._conn.execute(
            """
            SELECT id, 1.0 - (embedding <=> (SELECT embedding FROM concept WHERE id = %s)) AS sim
            FROM concept
            WHERE course = %s
              AND id <> %s
              AND embedding IS NOT NULL
              AND (SELECT embedding FROM concept WHERE id = %s) IS NOT NULL
            ORDER BY embedding <=> (SELECT embedding FROM concept WHERE id = %s), id
            LIMIT %s
            """,
            (concept_id, course, concept_id, concept_id, concept_id, k),
        )
        return [(row[0], row[1]) for row in cur.fetchall()]


class PostgresEdgeRepository(_PgRepo, EdgeRepository):
    """Knowledge-graph edges backed by the ``edge`` table (PK = src,dst,type)."""

    def upsert(self, edge: Edge) -> None:
        self._conn.execute(
            """
            INSERT INTO edge
                (src, dst, type, weight, importance, rationale, source_ref,
                 exposure_count, skip_count, last_traversed, provenance, confidence)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (src, dst, type) DO UPDATE SET
                weight = EXCLUDED.weight,
                importance = EXCLUDED.importance,
                rationale = EXCLUDED.rationale,
                source_ref = EXCLUDED.source_ref,
                exposure_count = EXCLUDED.exposure_count,
                skip_count = EXCLUDED.skip_count,
                last_traversed = EXCLUDED.last_traversed,
                provenance = EXCLUDED.provenance,
                confidence = EXCLUDED.confidence
            """,
            (
                edge.src,
                edge.dst,
                edge.type.value,
                edge.weight,
                edge.importance,
                edge.rationale,
                Jsonb(_ref_to_json(edge.source_ref))
                if edge.source_ref is not None
                else None,
                edge.exposure_count,
                edge.skip_count,
                edge.last_traversed,
                edge.provenance,
                edge.confidence,
            ),
        )

    def get(self, src: str, dst: str, type: EdgeType) -> Edge | None:
        cur = self._conn.execute(
            f"SELECT {_EDGE_COLS} FROM edge WHERE src = %s AND dst = %s AND type = %s",
            (src, dst, type.value),
        )
        row = cur.fetchone()
        return _row_to_edge(row) if row is not None else None

    def out_edges(self, src: str, type: EdgeType | None = None) -> Sequence[Edge]:
        if type is None:
            cur = self._conn.execute(
                f"SELECT {_EDGE_COLS} FROM edge WHERE src = %s "
                "ORDER BY src, type, dst",
                (src,),
            )
        else:
            cur = self._conn.execute(
                f"SELECT {_EDGE_COLS} FROM edge WHERE src = %s AND type = %s "
                "ORDER BY src, type, dst",
                (src, type.value),
            )
        return [_row_to_edge(r) for r in cur.fetchall()]

    def in_edges(self, dst: str, type: EdgeType | None = None) -> Sequence[Edge]:
        if type is None:
            cur = self._conn.execute(
                f"SELECT {_EDGE_COLS} FROM edge WHERE dst = %s "
                "ORDER BY src, type, dst",
                (dst,),
            )
        else:
            cur = self._conn.execute(
                f"SELECT {_EDGE_COLS} FROM edge WHERE dst = %s AND type = %s "
                "ORDER BY src, type, dst",
                (dst, type.value),
            )
        return [_row_to_edge(r) for r in cur.fetchall()]

    def list_by_course(self, course: str) -> Sequence[Edge]:
        # Edges carry no course; resolve through the SOURCE concept (the graph is
        # intra-course and directed, so src is authoritative) via a JOIN.
        cur = self._conn.execute(
            f"""
            SELECT {", ".join("e." + c for c in _EDGE_COLS.split(", "))}
            FROM edge e JOIN concept c ON e.src = c.id
            WHERE c.course = %s
            ORDER BY e.src, e.type, e.dst
            """,
            (course,),
        )
        return [_row_to_edge(r) for r in cur.fetchall()]

    def record_exposure(
        self, src: str, dst: str, type: EdgeType, *, skipped: bool, at: datetime
    ) -> None:
        """Account a traversal opportunity, upserting so it is total.

        Matches the in-memory adapter exactly: ``exposure_count`` always rises,
        ``skip_count`` rises only on a skip, and ``last_traversed`` advances only
        on an actual (non-skipped) traversal. We use INSERT .. ON CONFLICT DO
        UPDATE (not a bare UPDATE) so the accounting never fails just because
        ingestion has not created the edge yet (Liskov parity). The COALESCE
        keeps the prior ``last_traversed`` on a skip, because the inserted value
        is NULL exactly when ``skipped`` is True."""
        skip_increment = 1 if skipped else 0
        traversed_at = None if skipped else at
        self._conn.execute(
            """
            INSERT INTO edge (src, dst, type, exposure_count, skip_count, last_traversed)
            VALUES (%s, %s, %s, 1, %s, %s)
            ON CONFLICT (src, dst, type) DO UPDATE SET
                exposure_count = edge.exposure_count + 1,
                skip_count = edge.skip_count + EXCLUDED.skip_count,
                last_traversed = COALESCE(EXCLUDED.last_traversed, edge.last_traversed)
            """,
            (src, dst, type.value, skip_increment, traversed_at),
        )


class PostgresQuestionRepository(_PgRepo, QuestionRepository):
    """Question metadata backed by the ``question`` table (prose lives in OKF)."""

    def get(self, question_id: str) -> Question | None:
        cur = self._conn.execute(
            f"SELECT {_QUESTION_COLS} FROM question WHERE id = %s", (question_id,)
        )
        row = cur.fetchone()
        return _row_to_question(row) if row is not None else None

    def upsert(self, question: Question) -> None:
        self._conn.execute(
            """
            INSERT INTO question
                (id, concept_id, edge_id, kind, difficulty, hop_count,
                 source_refs, generated_by, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                concept_id = EXCLUDED.concept_id,
                edge_id = EXCLUDED.edge_id,
                kind = EXCLUDED.kind,
                difficulty = EXCLUDED.difficulty,
                hop_count = EXCLUDED.hop_count,
                source_refs = EXCLUDED.source_refs,
                generated_by = EXCLUDED.generated_by,
                status = EXCLUDED.status
            """,
            (
                question.id,
                question.concept_id,
                question.edge_id,
                question.kind,
                question.difficulty,
                question.hop_count,
                Jsonb(_refs_to_json(question.source_refs)),
                question.generated_by,
                question.status,
            ),
        )

    def by_concept(
        self,
        concept_id: str,
        *,
        difficulty: int | None = None,
        hop_count: int | None = None,
    ) -> Sequence[Question]:
        # Optional filters are independent exact matches (AND semantics). We use
        # "(%s IS NULL OR col = %s)" so one parameterised query covers every
        # combination without string-building the WHERE clause. Retired questions
        # are the kill switch: never served, so filtered out here.
        cur = self._conn.execute(
            f"""
            SELECT {_QUESTION_COLS} FROM question
            WHERE concept_id = %s
              AND status <> 'retired'
              AND (%s::int IS NULL OR difficulty = %s::int)
              AND (%s::int IS NULL OR hop_count = %s::int)
            ORDER BY id
            """,
            (concept_id, difficulty, difficulty, hop_count, hop_count),
        )
        return [_row_to_question(r) for r in cur.fetchall()]

    def by_edge(self, edge_id: str) -> Sequence[Question]:
        # Retired questions are excluded here too (the kill switch is global).
        cur = self._conn.execute(
            f"SELECT {_QUESTION_COLS} FROM question "
            "WHERE edge_id = %s AND status <> 'retired' ORDER BY id",
            (edge_id,),
        )
        return [_row_to_question(r) for r in cur.fetchall()]

    def retire(self, question_id: str) -> None:
        """Mark a question retired so it is never served again.

        Idempotent and total: a bare UPDATE matches zero rows when the id is
        unknown (a silent no-op, matching the in-memory adapter and ``get``'s
        tolerance of missing ids) and is harmless when already retired. The row
        survives (``get`` still returns it) but drops out of by_concept/by_edge."""
        self._conn.execute(
            "UPDATE question SET status = 'retired' WHERE id = %s", (question_id,)
        )


class PostgresLearnerStateRepository(_PgRepo, LearnerStateRepository):
    """FSRS state backed by ``learner_state`` (one row per concept)."""

    def get(self, concept_id: str) -> LearnerState | None:
        cur = self._conn.execute(
            f"SELECT {_STATE_COLS} FROM learner_state WHERE concept_id = %s",
            (concept_id,),
        )
        row = cur.fetchone()
        return _row_to_state(row) if row is not None else None

    def upsert(self, state: LearnerState) -> None:
        self._conn.execute(
            """
            INSERT INTO learner_state
                (concept_id, stability, difficulty, last_review, due_at,
                 reps, lapses, mastery)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (concept_id) DO UPDATE SET
                stability = EXCLUDED.stability,
                difficulty = EXCLUDED.difficulty,
                last_review = EXCLUDED.last_review,
                due_at = EXCLUDED.due_at,
                reps = EXCLUDED.reps,
                lapses = EXCLUDED.lapses,
                mastery = EXCLUDED.mastery
            """,
            (
                state.concept_id,
                state.stability,
                state.difficulty,
                state.last_review,
                state.due_at,
                state.reps,
                state.lapses,
                state.mastery.value,
            ),
        )

    def due(self, course: str, before: datetime) -> Sequence[LearnerState]:
        # No course column on learner_state -> JOIN concept to scope by course.
        # Only scheduled states (due_at set) that have come due by `before`;
        # boundary is inclusive (<=), matching the in-memory adapter.
        cur = self._conn.execute(
            f"""
            SELECT {", ".join("s." + c for c in _STATE_COLS.split(", "))}
            FROM learner_state s JOIN concept c ON s.concept_id = c.id
            WHERE c.course = %s AND s.due_at IS NOT NULL AND s.due_at <= %s
            ORDER BY s.due_at, s.concept_id
            """,
            (course, before),
        )
        return [_row_to_state(r) for r in cur.fetchall()]

    def all_for_course(self, course: str) -> Sequence[LearnerState]:
        cur = self._conn.execute(
            f"""
            SELECT {", ".join("s." + c for c in _STATE_COLS.split(", "))}
            FROM learner_state s JOIN concept c ON s.concept_id = c.id
            WHERE c.course = %s
            ORDER BY s.concept_id
            """,
            (course,),
        )
        return [_row_to_state(r) for r in cur.fetchall()]


class PostgresReviewLogRepository(_PgRepo, ReviewLogRepository):
    """Append-only review log backed by ``review_log`` (bigserial id)."""

    def append(self, event: ReviewEvent) -> None:
        self._conn.execute(
            """
            INSERT INTO review_log
                (concept_id, question_id, grade, fsrs_rating, predicted,
                 at, scheduler_ver)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                event.concept_id,
                event.question_id,
                event.grade,
                int(event.fsrs_rating),
                event.predicted,
                event.at,
                event.scheduler_ver,
            ),
        )

    def by_concept(self, concept_id: str) -> Sequence[ReviewEvent]:
        # ORDER BY the monotonically increasing serial id preserves append order
        # (the log is a per-concept time series), as the in-memory adapter does.
        cur = self._conn.execute(
            f"SELECT {_EVENT_COLS} FROM review_log WHERE concept_id = %s ORDER BY id",
            (concept_id,),
        )
        return [_row_to_event(r) for r in cur.fetchall()]


class PostgresTelemetryRepository(_PgRepo, TelemetryRepository):
    """Append-only engagement log backed by ``engagement_log`` (bigserial id)."""

    def append(self, event: EngagementEvent) -> None:
        self._conn.execute(
            """
            INSERT INTO engagement_log (kind, course, at, payload)
            VALUES (%s, %s, %s, %s)
            """,
            (
                event.kind,
                event.course,
                event.at,
                Jsonb(dict(event.payload)),
            ),
        )

    def last(self, kind: str, course: str) -> EngagementEvent | None:
        # Newest by `at`; the bigserial id breaks an exact `at` tie toward the
        # later-inserted row, matching the in-memory adapter's index tie-break.
        cur = self._conn.execute(
            f"""
            SELECT {_ENGAGEMENT_COLS} FROM engagement_log
            WHERE kind = %s AND course = %s
            ORDER BY at DESC, id DESC
            LIMIT 1
            """,
            (kind, course),
        )
        row = cur.fetchone()
        return _row_to_engagement(row) if row is not None else None

    def list_by_course(self, course: str) -> Sequence[EngagementEvent]:
        # ORDER BY the monotonic serial id preserves append order (the log is a
        # course-scoped time series), as the in-memory adapter does.
        cur = self._conn.execute(
            f"SELECT {_ENGAGEMENT_COLS} FROM engagement_log WHERE course = %s "
            "ORDER BY id",
            (course,),
        )
        return [_row_to_engagement(r) for r in cur.fetchall()]


class PostgresCourseProfileRepository(_PgRepo, CourseProfileRepository):
    """The one frozen profile per course, backed by ``course_profile``."""

    def get(self, course: str) -> CourseProfile | None:
        cur = self._conn.execute(
            f"SELECT {_PROFILE_COLS} FROM course_profile WHERE course = %s",
            (course,),
        )
        row = cur.fetchone()
        return _row_to_profile(row) if row is not None else None

    def upsert(self, profile: CourseProfile) -> None:
        self._conn.execute(
            """
            INSERT INTO course_profile
                (course, archetype, exam_format, weights, target_retention,
                 exam_date, confirmed_by_user)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (course) DO UPDATE SET
                archetype = EXCLUDED.archetype,
                exam_format = EXCLUDED.exam_format,
                weights = EXCLUDED.weights,
                target_retention = EXCLUDED.target_retention,
                exam_date = EXCLUDED.exam_date,
                confirmed_by_user = EXCLUDED.confirmed_by_user
            """,
            (
                profile.course,
                profile.archetype,
                Jsonb(dict(profile.exam_format)),
                Jsonb(dict(profile.weights)),
                profile.target_retention,
                profile.exam_date,
                profile.confirmed_by_user,
            ),
        )


class PostgresRepositories:
    """Container wiring one connection to all Postgres-owned repositories.

    Registering the pgvector type on the connection at construction time means
    the ``vector`` OID is known for the lifetime of every adapter that shares
    this connection. The registration is guarded so the container can also be
    constructed in driver-less smoke tests (no SQL runs until a method is
    called). Content prose is intentionally absent here: it is OKF-owned and
    served by a separate ContentRepository, per the OKF/Postgres split."""

    def __init__(self, conn) -> None:
        if register_vector is not None:
            register_vector(conn)
        self._conn = conn
        self.concepts = PostgresConceptIndexRepository(conn)
        self.edges = PostgresEdgeRepository(conn)
        self.questions = PostgresQuestionRepository(conn)
        self.learner_state = PostgresLearnerStateRepository(conn)
        self.review_log = PostgresReviewLogRepository(conn)
        self.telemetry = PostgresTelemetryRepository(conn)
        self.profiles = PostgresCourseProfileRepository(conn)


def connect(database_url: str):
    """Open a psycopg connection to ``database_url`` (guarded on the driver).

    Kept minimal and single-responsibility: it only opens the connection. Pass
    the result to ``PostgresRepositories``, which performs the pgvector type
    registration. Raising a clear RuntimeError when the driver is missing turns
    an obscure ``AttributeError: 'NoneType'`` into an actionable message."""
    if psycopg is None:
        raise RuntimeError(
            "psycopg is not installed; install 'psycopg[binary]' and 'pgvector' "
            "to use the Postgres adapter"
        )
    # autocommit: operations are mostly single statements, so committing each one
    # independently avoids a stray error poisoning the whole connection (the
    # aborted-transaction cascade) and makes writes durable without an explicit
    # commit per call.
    return psycopg.connect(database_url, autocommit=True)

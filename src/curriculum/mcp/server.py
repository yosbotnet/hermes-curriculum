"""Stdio MCP server: a thin transport adapter over the CurriculumService port.

This is the outermost ring of the hexagon. It owns NO business logic: every
tool handler does exactly two things -- call the matching ``CurriculumService``
method and serialise the returned domain DTO to a plain JSON-able ``dict`` -- so
that the same use-cases an MCP client drives are byte-for-byte the ones the unit
tests drive directly. Keeping the (de)serialisation here, behind small pure
``*_to_dict`` helpers and pure ``_call_*`` routers, means the meaning of each
tool is fully testable WITHOUT the optional ``mcp`` package installed.

Optional-dependency guard
-------------------------
The ``mcp`` SDK is an optional, heavyweight dependency. Mirroring the Postgres
adapter, we import it inside a ``try`` and fall back to ``None`` so this module
(and the rest of the package, its tests, and offline runs) imports cleanly on a
machine with no ``mcp`` installed. ``build_server`` and ``main`` raise/print a
clear message rather than dereferencing a ``None`` sentinel in anger.
"""
from __future__ import annotations

import json
import sys
from typing import Any, Sequence

from ..config import load as load_settings
from ..domain.entities import (
    ConceptContent,
    NextAction,
    NextResult,
    Question,
    QuestionContent,
    ScoredCandidate,
    SourceRef,
)
from ..ports.service import CurriculumService

# --------------------------------------------------------------------------- #
# Optional-driver guard: the module imports cleanly even with no mcp installed.
# We bind ``mcp`` (the package) and ``FastMCP`` (the high-level server used to
# register tools and serve stdio). If the import fails we disable the whole
# adapter by setting both symbols to None; tests then skipUnless(mcp is not None)
# and the two entrypoints fail loudly with an actionable message.
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - exercised only where the SDK is installed
    import mcp
    from mcp.server.fastmcp import FastMCP
except ImportError:  # pragma: no cover - the import-without-SDK path
    mcp = None
    FastMCP = None

__all__ = [
    "TOOL_NAMES",
    "build_server",
    "main",
    "next_result_to_dict",
    "next_action_to_dict",
    "scored_candidate_to_dict",
    "concept_content_to_dict",
    "question_to_dict",
    "question_content_to_dict",
    "quiz_to_dict",
]

# The stable public name of every tool this server exposes. Kept as data so the
# wiring and the tests assert against one source of truth.
TOOL_NAMES: tuple[str, ...] = (
    "next",
    "explain",
    "quiz",
    "grade",
    "state",
    "courses",
    "checkin",
    "frontier",
    "flag_question",
)


# --------------------------------------------------------------------------- #
# Pure serialisation helpers (no mcp, no I/O) -- the actual contract of each
# tool's wire shape, unit-testable on their own.
# --------------------------------------------------------------------------- #
def _source_ref_to_dict(ref: SourceRef) -> dict[str, Any]:
    """A grounding pointer becomes ``{file, line}``; ``line`` may be JSON null."""
    return {"file": ref.file, "line": ref.line}


def _source_refs_to_list(refs: Sequence[SourceRef]) -> list[dict[str, Any]]:
    return [_source_ref_to_dict(r) for r in refs]


def next_action_to_dict(action: NextAction) -> dict[str, Any]:
    """Serialise the chosen action; enums are emitted as their string values so
    the payload needs no custom JSON encoder on the client side."""
    return {
        "mode": action.mode.value,
        "concept_id": action.concept_id,
        "reason": action.reason,
        "source_refs": _source_refs_to_list(action.source_refs),
        "question_id": action.question_id,
    }


def scored_candidate_to_dict(candidate: ScoredCandidate) -> dict[str, Any]:
    return {
        "concept_id": candidate.concept_id,
        "mode": candidate.mode.value,
        "score": candidate.score,
    }


def next_result_to_dict(result: NextResult) -> dict[str, Any]:
    """``NextResult`` -> ``{chosen, candidates, temperature}``.

    The whole ranked field and the sampling temperature are exposed (not just
    the winner) so a client can inspect or override the engine's choice."""
    return {
        "chosen": next_action_to_dict(result.chosen),
        "candidates": [scored_candidate_to_dict(c) for c in result.candidates],
        "temperature": result.temperature,
    }


def concept_content_to_dict(content: ConceptContent) -> dict[str, Any]:
    return {
        "concept_id": content.concept_id,
        "title": content.title,
        "body": content.body,
        "description": content.description,
        "source_refs": _source_refs_to_list(content.source_refs),
    }


def question_to_dict(question: Question) -> dict[str, Any]:
    return {
        "id": question.id,
        "concept_id": question.concept_id,
        "kind": question.kind,
        "difficulty": question.difficulty,
        "hop_count": question.hop_count,
        "edge_id": question.edge_id,
        "source_refs": _source_refs_to_list(question.source_refs),
        "generated_by": question.generated_by,
    }


def question_content_to_dict(content: QuestionContent) -> dict[str, Any]:
    return {
        "question_id": content.question_id,
        "prompt": content.prompt,
        "rubric": content.rubric,
    }


def quiz_to_dict(pair: tuple[Question, QuestionContent]) -> dict[str, Any]:
    """``quiz`` returns metadata + prose; the client wants both, keyed apart so
    the structural id/difficulty stay distinct from the prompt/rubric text."""
    question, content = pair
    return {
        "question": question_to_dict(question),
        "content": question_content_to_dict(content),
    }


# --------------------------------------------------------------------------- #
# Pure routers (no mcp): one per tool, each calls the service and serialises.
# These are the transport-agnostic core; build_server only wraps them with a
# typed signature so the SDK can derive each tool's JSON input schema.
# --------------------------------------------------------------------------- #
def _call_next(
    service: CurriculumService, course: str, focus: str | None = None
) -> dict[str, Any]:
    return next_result_to_dict(service.next_action(course, focus=focus))


def _call_explain(service: CurriculumService, concept_id: str) -> dict[str, Any]:
    return concept_content_to_dict(service.explain(concept_id))


def _call_quiz(
    service: CurriculumService, concept_id: str, *, difficulty: int | None = None
) -> dict[str, Any]:
    return quiz_to_dict(service.quiz(concept_id, difficulty=difficulty))


def _call_grade(
    service: CurriculumService,
    *,
    concept_id: str,
    score: int,
    question_id: str | None = None,
    predicted: int | None = None,
    traversed_edges: Sequence[str] | None = None,
    skipped_edges: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Bridge the wire shape to the port: JSON arrays arrive as lists, but the
    service signature wants ``tuple[str, ...]`` (the DTOs are frozen/hashable),
    so we coerce here -- and ``None`` (an omitted optional array) to the empty
    tuple, matching the port's default."""
    result = service.grade(
        concept_id=concept_id,
        score=score,
        question_id=question_id,
        predicted=predicted,
        traversed_edges=tuple(traversed_edges or ()),
        skipped_edges=tuple(skipped_edges or ()),
    )
    return dict(result)


def _call_state(service: CurriculumService, course: str) -> dict[str, Any]:
    return dict(service.state(course))


def _call_checkin(service: CurriculumService, course: str) -> dict[str, Any]:
    """The honest game-state reading. The service payload is already a plain
    JSON-able mapping, so the router only coerces the outer Mapping to a dict."""
    return dict(service.checkin(course))


def _call_frontier(
    service: CurriculumService, course: str, focus: str | None = None
) -> dict[str, Any]:
    """The strategy buckets for a course, optionally scoped by ``focus``. The
    payload is a JSON-able mapping of buckets; coerce only the outer Mapping."""
    return dict(service.frontier(course, focus=focus))


def _call_flag_question(
    service: CurriculumService, question_id: str, reason: str = ""
) -> dict[str, Any]:
    """The item kill switch: retire a question. Returns the JSON-able status
    mapping, coerced to a plain dict for the wire."""
    return dict(service.flag_question(question_id, reason=reason))


# --------------------------------------------------------------------------- #
# MCP wiring (the only part that touches the SDK).
# --------------------------------------------------------------------------- #
def build_server(service: CurriculumService) -> Any:
    """Build a FastMCP server exposing ``service`` as the eight curriculum tools.

    Each registered handler is a thin closure over ``service`` that defers to a
    pure ``_call_*`` router (above). The return annotations are deliberately
    omitted on the handlers: we want the SDK to emit the JSON serialisation as
    plain text content without inferring an output schema, keeping the wire
    payload identical across SDK versions and matching the ``_call_*`` dicts the
    unit tests assert on.
    """
    if FastMCP is None:
        raise RuntimeError(
            "the 'mcp' package is not installed; install 'mcp' to build the "
            "curriculum MCP server"
        )

    server = FastMCP("curriculum")

    @server.tool(
        name="next",
        description=(
            "Decide the single best next action for a course (teach/review/"
            "test) and return it with the ranked candidate field and the "
            "sampling temperature. Optional 'focus' scopes the choice to one "
            "topic/module: comma/space-separated substrings matched against "
            "concept ids and source tokens (e.g. 'crypto', 'cyber-03', 'm2'). "
            "Call 'state' to see the available topics."
        ),
    )
    def next_tool(course: str, focus: str | None = None):  # noqa: ANN202 - schema is derived from params
        return _call_next(service, course, focus)

    @server.tool(
        name="explain",
        description="Return grounded teaching prose for a concept.",
    )
    def explain_tool(concept_id: str):  # noqa: ANN202
        return _call_explain(service, concept_id)

    @server.tool(
        name="quiz",
        description=(
            "Pick a question for a concept (optionally at a target difficulty) "
            "and return its metadata plus prompt/rubric text."
        ),
    )
    def quiz_tool(concept_id: str, difficulty: int | None = None):  # noqa: ANN202
        return _call_quiz(service, concept_id, difficulty=difficulty)

    @server.tool(
        name="grade",
        description=(
            "Record a graded answer: updates the spaced-repetition schedule, "
            "runs FIRe propagation, updates connection-skip counts, and logs "
            "calibration. Returns the new schedule."
        ),
    )
    def grade_tool(  # noqa: ANN202
        concept_id: str,
        score: int,
        question_id: str | None = None,
        predicted: int | None = None,
        traversed_edges: list[str] | None = None,
        skipped_edges: list[str] | None = None,
    ):
        return _call_grade(
            service,
            concept_id=concept_id,
            score=score,
            question_id=question_id,
            predicted=predicted,
            traversed_edges=traversed_edges,
            skipped_edges=skipped_edges,
        )

    @server.tool(
        name="state",
        description="Return a burndown / progress snapshot for a course.",
    )
    def state_tool(course: str):  # noqa: ANN202
        return _call_state(service, course)

    @server.tool(
        name="courses",
        description="List all courses available in the curriculum engine.",
    )
    def courses_tool():  # noqa: ANN202
        return {"courses": service.list_courses()}

    @server.tool(
        name="checkin",
        description=(
            "Return the honest game-state reading for a course: importance-"
            "weighted memory capital (stability_days), its delta since the last "
            "check, consolidation and ripeness, the concepts ready to unlock, "
            "the near-unlocks, and the mastery breakdown. Logs one 'check' "
            "engagement event so the next check can diff against it."
        ),
    )
    def checkin_tool(course: str):  # noqa: ANN202
        return _call_checkin(service, course)

    @server.tool(
        name="frontier",
        description=(
            "Return up to three strategy buckets for what to pursue next: 'push' "
            "(the best new concept to start), 'reinforce' (the review with the "
            "weakest recall right now) and 'breakthrough' (the nearest locked "
            "concept). Optional 'focus' scopes the buckets to one topic/module. "
            "A pure read: it consumes no candidate and advances nothing -- "
            "choosing happens later via next/quiz."
        ),
    )
    def frontier_tool(course: str, focus: str | None = None):  # noqa: ANN202
        return _call_frontier(service, course, focus)

    @server.tool(
        name="flag_question",
        description=(
            "Retire a question so it is never served again (the item kill "
            "switch), logging an 'item_flag' engagement event with the optional "
            "reason. Returns the question id and its 'retired' status."
        ),
    )
    def flag_question_tool(question_id: str, reason: str = ""):  # noqa: ANN202
        return _call_flag_question(service, question_id, reason)

    return server


def main(argv: Sequence[str] | None = None) -> int:
    """Console entrypoint: build the real service and serve over stdio.

    Returns a process exit code so the wiring is testable without raising
    ``SystemExit`` mid-suite (the ``__main__`` guard converts it to one).

    Order of guards is intentional: if the SDK is absent there is nothing to
    run, so we say so and bail before touching config or composition. The
    composition layer (the wiring of repositories + strategies into the service)
    is imported lazily inside a ``try`` because it may not exist yet / may not be
    installed in every deployment; a clear message beats an opaque ImportError.
    """
    if mcp is None:
        print(
            "the 'mcp' package is not installed; install 'mcp' to run the "
            "curriculum MCP server",
            file=sys.stderr,
        )
        return 1

    settings = load_settings()
    try:
        from curriculum.application.composition import build_service
    except ImportError:
        print(
            "the application composition layer "
            "(curriculum.application.composition) is unavailable; cannot build "
            "the curriculum service",
            file=sys.stderr,
        )
        return 1

    service = build_service(settings)
    server = build_server(service)
    # FastMCP.run() defaults to the stdio transport (mcp.server.stdio): it owns
    # the event loop and the read/write streams for the lifetime of the process.
    server.run()
    return 0


if __name__ == "__main__":  # pragma: no cover - process entrypoint
    sys.exit(main())

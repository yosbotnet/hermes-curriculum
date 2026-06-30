"""Tests for the stdio MCP server adapter.

Two layers, mirroring the Postgres adapter's pattern:

1. Always-on tests: the module MUST import with no ``mcp`` SDK installed, the
   pure ``*_to_dict`` serialisers must produce the agreed wire shapes, the pure
   ``_call_*`` routers must call the right service method and coerce the wire
   types (notably JSON arrays -> tuples for ``grade``), and the two entrypoints
   must fail loudly (not with an AttributeError) when the SDK is absent. These
   pin the transport contract without needing the optional dependency.

2. SDK-gated tests (skipUnless ``mcp`` is installed): build the real FastMCP
   server over a stub service, assert the five tools are registered, and assert
   a routed call returns exactly the serialised dict.
"""
from __future__ import annotations

import asyncio
import io
import json
import unittest
from contextlib import redirect_stderr

from curriculum.domain.entities import (
    ConceptContent,
    NextAction,
    NextResult,
    Question,
    QuestionContent,
    ScoredCandidate,
    SourceRef,
)
from curriculum.domain.enums import NextMode
from curriculum.mcp import server
from curriculum.mcp.server import mcp
from curriculum.ports.service import CurriculumService


# --------------------------------------------------------------------------- #
# A deterministic stub service returning canned DTOs. It is a real subclass of
# the port (Liskov) so the same object drives the pure routers and the SDK.
# --------------------------------------------------------------------------- #
class _StubService(CurriculumService):
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def next_action(self, course: str) -> NextResult:
        self.calls.append(("next_action", course))
        return NextResult(
            chosen=NextAction(
                mode=NextMode.TEACH,
                concept_id="c1",
                reason="new learnable concept",
                source_refs=(SourceRef(file="intro.md", line=1),),
                question_id=None,
            ),
            candidates=(
                ScoredCandidate(concept_id="c1", mode=NextMode.TEACH, score=1.5),
                ScoredCandidate(concept_id="c2", mode=NextMode.REVIEW, score=0.5),
            ),
            temperature=0.6,
        )

    def explain(self, concept_id: str) -> ConceptContent:
        self.calls.append(("explain", concept_id))
        return ConceptContent(
            concept_id=concept_id,
            title="Title",
            body="body text",
            description="a short description",
            source_refs=(SourceRef(file="intro.md", line=2),),
        )

    def quiz(self, concept_id: str, *, difficulty: int | None = None):
        self.calls.append(("quiz", concept_id, difficulty))
        question = Question(
            id="q1",
            concept_id=concept_id,
            kind="open",
            difficulty=difficulty or 1,
            hop_count=1,
            edge_id=None,
            source_refs=(SourceRef(file="intro.md", line=3),),
            generated_by="fake",
        )
        content = QuestionContent(question_id="q1", prompt="What is X?", rubric="mention Y")
        return question, content

    def grade(
        self,
        *,
        concept_id: str,
        score: int,
        question_id: str | None = None,
        predicted: int | None = None,
        traversed_edges: tuple[str, ...] = (),
        skipped_edges: tuple[str, ...] = (),
    ):
        self.calls.append(
            (
                "grade",
                concept_id,
                score,
                question_id,
                predicted,
                traversed_edges,
                skipped_edges,
            )
        )
        return {
            "concept_id": concept_id,
            "rating": 3,
            "mastery": "learning",
            "due_at": "2026-07-01T00:00:00",
            "stability": 5.0,
            "fire_credits": [],
            "escalated_connections": [],
        }

    def state(self, course: str):
        self.calls.append(("state", course))
        return {
            "course": course,
            "total": 3,
            "by_mastery": {"new": 1, "learning": 1, "solid": 1, "exam_ready": 0},
            "due_now": 1,
        }


def _payload(raw):
    """Extract the JSON dict from whatever ``FastMCP.call_tool`` returned.

    The SDK's return shape evolved across versions: older builds return just the
    content sequence, newer ones return ``(content, structured_content)``. The
    one thing every version emits is a TextContent block carrying the JSON, so
    we normalise to the content sequence and parse the first block's text."""
    content = raw
    if isinstance(raw, tuple) and len(raw) == 2 and isinstance(raw[0], (list, tuple)):
        content = raw[0]
    for block in content:
        text = getattr(block, "text", None)
        if text is not None:
            return json.loads(text)
    raise AssertionError("tool result carried no text content")


# --------------------------------------------------------------------------- #
# Always-on: import contract + pure serialisers + pure routers.
# --------------------------------------------------------------------------- #
class ImportContractTest(unittest.TestCase):
    def test_module_imports_and_exposes_surface(self) -> None:
        # The top-level import already proves the module loads with or without
        # the SDK; assert the public surface and the guarded sentinel exist.
        self.assertTrue(hasattr(server, "build_server"))
        self.assertTrue(hasattr(server, "main"))
        self.assertTrue(hasattr(server, "mcp"))  # None when the SDK is absent
        self.assertEqual(server.TOOL_NAMES, ("next", "explain", "quiz", "grade", "state"))


class SerialiserTest(unittest.TestCase):
    def setUp(self) -> None:
        self.stub = _StubService()

    def test_next_result_to_dict_shape(self) -> None:
        out = server.next_result_to_dict(self.stub.next_action("Cyber"))
        self.assertEqual(out["temperature"], 0.6)
        self.assertEqual(out["chosen"]["concept_id"], "c1")
        self.assertEqual(out["chosen"]["mode"], "teach")  # enum -> string value
        self.assertEqual(out["chosen"]["source_refs"], [{"file": "intro.md", "line": 1}])
        self.assertIsNone(out["chosen"]["question_id"])
        self.assertEqual(
            out["candidates"],
            [
                {"concept_id": "c1", "mode": "teach", "score": 1.5},
                {"concept_id": "c2", "mode": "review", "score": 0.5},
            ],
        )

    def test_next_result_is_json_serialisable(self) -> None:
        # The whole point of the adapter: the payload survives a JSON round-trip.
        out = server.next_result_to_dict(self.stub.next_action("Cyber"))
        self.assertEqual(json.loads(json.dumps(out)), out)

    def test_concept_content_to_dict_shape(self) -> None:
        out = server.concept_content_to_dict(self.stub.explain("c1"))
        self.assertEqual(
            out,
            {
                "concept_id": "c1",
                "title": "Title",
                "body": "body text",
                "description": "a short description",
                "source_refs": [{"file": "intro.md", "line": 2}],
            },
        )

    def test_quiz_to_dict_keys_metadata_apart_from_prose(self) -> None:
        out = server.quiz_to_dict(self.stub.quiz("c1", difficulty=2))
        self.assertEqual(out["question"]["id"], "q1")
        self.assertEqual(out["question"]["difficulty"], 2)
        self.assertEqual(out["question"]["source_refs"], [{"file": "intro.md", "line": 3}])
        self.assertEqual(out["content"], {"question_id": "q1", "prompt": "What is X?", "rubric": "mention Y"})

    def test_source_ref_none_line_preserved(self) -> None:
        ref = SourceRef(file="f.md", line=None)
        self.assertEqual(server._source_ref_to_dict(ref), {"file": "f.md", "line": None})


class PureRouterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.stub = _StubService()

    def test_call_next_routes_and_serialises(self) -> None:
        out = server._call_next(self.stub, "Cyber")
        self.assertEqual(self.stub.calls[-1], ("next_action", "Cyber"))
        self.assertEqual(out["chosen"]["concept_id"], "c1")

    def test_call_explain_routes(self) -> None:
        out = server._call_explain(self.stub, "c9")
        self.assertEqual(self.stub.calls[-1], ("explain", "c9"))
        self.assertEqual(out["concept_id"], "c9")

    def test_call_quiz_forwards_difficulty(self) -> None:
        server._call_quiz(self.stub, "c1", difficulty=4)
        self.assertEqual(self.stub.calls[-1], ("quiz", "c1", 4))

    def test_call_quiz_default_difficulty_is_none(self) -> None:
        server._call_quiz(self.stub, "c1")
        self.assertEqual(self.stub.calls[-1], ("quiz", "c1", None))

    def test_call_grade_coerces_arrays_to_tuples(self) -> None:
        out = server._call_grade(
            self.stub,
            concept_id="c1",
            score=4,
            traversed_edges=["e1"],
            skipped_edges=["e2", "e3"],
        )
        # The port wants hashable tuples; lists from JSON must be coerced.
        self.assertEqual(self.stub.calls[-1][5], ("e1",))
        self.assertEqual(self.stub.calls[-1][6], ("e2", "e3"))
        self.assertEqual(out["concept_id"], "c1")

    def test_call_grade_none_arrays_become_empty_tuples(self) -> None:
        server._call_grade(self.stub, concept_id="c1", score=2)
        self.assertEqual(self.stub.calls[-1][5], ())
        self.assertEqual(self.stub.calls[-1][6], ())

    def test_call_state_routes_and_is_plain_dict(self) -> None:
        out = server._call_state(self.stub, "Cyber")
        self.assertEqual(self.stub.calls[-1], ("state", "Cyber"))
        self.assertIsInstance(out, dict)
        self.assertEqual(out["course"], "Cyber")


# --------------------------------------------------------------------------- #
# Guard behaviour when the SDK is absent (only meaningful without it).
# --------------------------------------------------------------------------- #
@unittest.skipUnless(mcp is None, "guard behaviour only applies without the SDK")
class MissingSdkGuardTest(unittest.TestCase):
    def test_build_server_raises_clear_error(self) -> None:
        with self.assertRaises(RuntimeError):
            server.build_server(_StubService())

    def test_main_reports_and_exits_nonzero(self) -> None:
        err = io.StringIO()
        with redirect_stderr(err):
            code = server.main()
        self.assertEqual(code, 1)
        self.assertIn("mcp", err.getvalue().lower())


# --------------------------------------------------------------------------- #
# SDK-gated: real FastMCP wiring + routed serialisation.
# --------------------------------------------------------------------------- #
@unittest.skipUnless(mcp is not None, "needs the mcp SDK installed")
class McpServerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.stub = _StubService()
        self.server = server.build_server(self.stub)

    def test_build_server_returns_an_object(self) -> None:
        self.assertIsNotNone(self.server)

    def test_all_five_tools_registered(self) -> None:
        tools = asyncio.run(self.server.list_tools())
        self.assertEqual({t.name for t in tools}, set(server.TOOL_NAMES))

    def test_every_tool_has_an_input_schema(self) -> None:
        tools = asyncio.run(self.server.list_tools())
        for tool in tools:
            self.assertIsInstance(tool.inputSchema, dict)
            self.assertTrue(tool.inputSchema, f"{tool.name} has an empty input schema")

    def test_routed_explain_returns_serialised_dict(self) -> None:
        raw = asyncio.run(self.server.call_tool("explain", {"concept_id": "c1"}))
        self.assertEqual(_payload(raw), server.concept_content_to_dict(self.stub.explain("c1")))

    def test_routed_next_exposes_ranked_field(self) -> None:
        raw = asyncio.run(self.server.call_tool("next", {"course": "Cyber"}))
        payload = _payload(raw)
        self.assertEqual(payload["chosen"]["concept_id"], "c1")
        self.assertEqual(payload["chosen"]["mode"], "teach")
        self.assertEqual(len(payload["candidates"]), 2)
        self.assertAlmostEqual(payload["temperature"], 0.6)

    def test_routed_quiz_forwards_difficulty(self) -> None:
        raw = asyncio.run(self.server.call_tool("quiz", {"concept_id": "c1", "difficulty": 3}))
        payload = _payload(raw)
        self.assertEqual(payload["question"]["difficulty"], 3)
        self.assertEqual(self.stub.calls[-1], ("quiz", "c1", 3))

    def test_routed_grade_coerces_arrays_to_tuples(self) -> None:
        raw = asyncio.run(
            self.server.call_tool(
                "grade",
                {
                    "concept_id": "c1",
                    "score": 4,
                    "traversed_edges": ["e1"],
                    "skipped_edges": ["e2", "e3"],
                },
            )
        )
        payload = _payload(raw)
        self.assertEqual(payload["concept_id"], "c1")
        grade_calls = [c for c in self.stub.calls if c[0] == "grade"]
        self.assertEqual(grade_calls[-1][5], ("e1",))
        self.assertEqual(grade_calls[-1][6], ("e2", "e3"))

    def test_routed_state_returns_snapshot(self) -> None:
        raw = asyncio.run(self.server.call_tool("state", {"course": "Cyber"}))
        payload = _payload(raw)
        self.assertEqual(payload["course"], "Cyber")
        self.assertEqual(payload["total"], 3)


if __name__ == "__main__":
    unittest.main()

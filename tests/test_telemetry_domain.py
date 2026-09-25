"""Domain-level tests for the motivation layer foundation.

Covers the new frozen telemetry event, the edge provenance/confidence fields,
and the question status kill-switch field. These are pure value-object tests:
construct, check defaults, and assert immutability. Written with unittest (not
pytest) so ``make test`` -- stdlib unittest discovery -- can import it.
"""
from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

from curriculum.domain.entities import Edge, Question
from curriculum.domain.enums import EdgeType
from curriculum.domain.telemetry import EngagementEvent

AT = datetime(2026, 7, 3, 12, 0, tzinfo=timezone.utc)


class EngagementEventTests(unittest.TestCase):
    def test_defaults_and_fields(self) -> None:
        event = EngagementEvent(kind="check", course="phys101", at=AT)
        self.assertEqual(event.kind, "check")
        self.assertEqual(event.course, "phys101")
        self.assertEqual(event.at, AT)
        self.assertEqual(event.payload, {})

    def test_payload_carried(self) -> None:
        event = EngagementEvent(kind="item_flag", course="phys101", at=AT, payload={"question_id": "q1"})
        self.assertEqual(event.payload, {"question_id": "q1"})

    def test_is_frozen(self) -> None:
        event = EngagementEvent(kind="check", course="phys101", at=AT)
        with self.assertRaises(FrozenInstanceError):
            event.kind = "escalate"  # type: ignore[misc]


class MotivationFieldDefaultsTests(unittest.TestCase):
    def test_edge_provenance_and_confidence_defaults(self) -> None:
        edge = Edge(src="a", dst="b", type=EdgeType.PREREQUISITE)
        self.assertEqual(edge.provenance, "inferred")
        self.assertEqual(edge.confidence, 0.6)

    def test_question_status_default_active(self) -> None:
        self.assertEqual(Question(id="q", concept_id="c").status, "active")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

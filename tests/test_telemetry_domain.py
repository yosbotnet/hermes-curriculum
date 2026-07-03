"""Domain-level tests for the motivation layer foundation.

Covers the new frozen telemetry event, the edge provenance/confidence fields,
and the question status kill-switch field. These are pure value-object tests:
construct, check defaults, and assert immutability.
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pytest

from curriculum.domain.entities import Edge, Question
from curriculum.domain.enums import EdgeType
from curriculum.domain.telemetry import EngagementEvent


def test_engagement_event_defaults_and_fields() -> None:
    at = datetime(2026, 7, 3, 12, 0, tzinfo=timezone.utc)
    event = EngagementEvent(kind="check", course="phys101", at=at)
    assert event.kind == "check"
    assert event.course == "phys101"
    assert event.at == at
    assert event.payload == {}


def test_engagement_event_payload_carried() -> None:
    at = datetime(2026, 7, 3, 12, 0, tzinfo=timezone.utc)
    event = EngagementEvent(
        kind="item_flag", course="phys101", at=at, payload={"question_id": "q1"}
    )
    assert event.payload == {"question_id": "q1"}


def test_engagement_event_is_frozen() -> None:
    at = datetime(2026, 7, 3, 12, 0, tzinfo=timezone.utc)
    event = EngagementEvent(kind="check", course="phys101", at=at)
    with pytest.raises(FrozenInstanceError):
        event.kind = "escalate"  # type: ignore[misc]


def test_edge_provenance_and_confidence_defaults() -> None:
    edge = Edge(src="a", dst="b", type=EdgeType.PREREQUISITE)
    assert edge.provenance == "inferred"
    assert edge.confidence == 0.6


def test_question_status_default_active() -> None:
    q = Question(id="q", concept_id="c")
    assert q.status == "active"

"""Engagement telemetry (the motivation layer's append-only signal).

A single frozen, slotted value object records that something happened for a
course at a point in time. Like `ReviewEvent`, these are append-only facts,
never mutated; the authoritative home is Postgres (the `engagement_log` table).
The loosely-typed `payload` keeps the schema stable while letting different
event kinds carry their own detail (e.g. a flagged question id).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class EngagementEvent:
    """An append-only record of one engagement signal for a course."""

    kind: str                                 # check | escalate | session_start | session_end | item_flag
    course: str
    at: datetime
    payload: Mapping[str, Any] = field(default_factory=dict)

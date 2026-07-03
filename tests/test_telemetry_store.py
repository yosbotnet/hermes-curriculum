"""Tests for the in-memory engagement telemetry adapter.

Stdlib unittest only. The telemetry store is the append-only signal that feeds
the motivation layer, so the invariants under test are: append/list preserves
what was recorded, ``last(kind, course)`` returns the most recent event by its
``at`` timestamp (with the latest-appended winning an exact ``at`` tie, mirroring
the Postgres ``ORDER BY at DESC, id DESC LIMIT 1``), and both ``last`` and
``list_by_course`` filter strictly by kind/course.
"""
from __future__ import annotations

import unittest
from datetime import datetime, timedelta

from curriculum.domain.telemetry import EngagementEvent
from curriculum.ports.repositories import TelemetryRepository
from curriculum.storage.memory import InMemoryTelemetryRepository

T0 = datetime(2026, 1, 1, 12, 0, 0)


class InMemoryTelemetryRepositoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.repo = InMemoryTelemetryRepository()

    def _event(
        self, kind: str, course: str, at: datetime, **payload
    ) -> EngagementEvent:
        return EngagementEvent(kind=kind, course=course, at=at, payload=payload)

    def test_is_a_telemetry_repository(self) -> None:
        self.assertIsInstance(self.repo, TelemetryRepository)

    def test_append_then_list_by_course_round_trip(self) -> None:
        e = self._event("check", "cs101", T0, question_id="q1")
        self.repo.append(e)
        self.assertEqual(self.repo.list_by_course("cs101"), [e])

    def test_list_by_course_empty_when_none(self) -> None:
        self.assertEqual(self.repo.list_by_course("cs101"), [])

    def test_list_by_course_filters_by_course(self) -> None:
        a = self._event("check", "cs101", T0)
        b = self._event("check", "other", T0 + timedelta(hours=1))
        self.repo.append(a)
        self.repo.append(b)
        self.assertEqual(self.repo.list_by_course("cs101"), [a])

    def test_list_by_course_preserves_append_order(self) -> None:
        e1 = self._event("check", "cs101", T0)
        e2 = self._event("escalate", "cs101", T0 + timedelta(hours=2))
        e3 = self._event("check", "cs101", T0 + timedelta(hours=1))
        for e in (e1, e2, e3):
            self.repo.append(e)
        # Append order preserved verbatim (the log is a time series as written).
        self.assertEqual(self.repo.list_by_course("cs101"), [e1, e2, e3])

    def test_last_returns_newest_by_at(self) -> None:
        oldest = self._event("check", "cs101", T0)
        middle = self._event("check", "cs101", T0 + timedelta(hours=1))
        newest = self._event("check", "cs101", T0 + timedelta(hours=2))
        # Append out of chronological order: last() must sort by `at`, not by
        # insertion order, and return the newest.
        self.repo.append(middle)
        self.repo.append(newest)
        self.repo.append(oldest)
        self.assertEqual(self.repo.last("check", "cs101"), newest)

    def test_last_filters_by_kind(self) -> None:
        check = self._event("check", "cs101", T0)
        escalate = self._event("escalate", "cs101", T0 + timedelta(hours=1))
        self.repo.append(check)
        self.repo.append(escalate)
        # A newer event of a different kind must not shadow the check.
        self.assertEqual(self.repo.last("check", "cs101"), check)

    def test_last_filters_by_course(self) -> None:
        here = self._event("check", "cs101", T0)
        elsewhere = self._event("check", "other", T0 + timedelta(hours=1))
        self.repo.append(here)
        self.repo.append(elsewhere)
        self.assertEqual(self.repo.last("check", "cs101"), here)

    def test_last_missing_returns_none(self) -> None:
        self.repo.append(self._event("check", "cs101", T0))
        self.assertIsNone(self.repo.last("escalate", "cs101"))
        self.assertIsNone(self.repo.last("check", "other"))

    def test_last_breaks_at_ties_toward_latest_appended(self) -> None:
        first = self._event("check", "cs101", T0, seq=1)
        second = self._event("check", "cs101", T0, seq=2)
        self.repo.append(first)
        self.repo.append(second)
        # Same `at`: the later-appended wins (mirrors Postgres id DESC tie-break).
        self.assertEqual(self.repo.last("check", "cs101"), second)


if __name__ == "__main__":
    unittest.main()

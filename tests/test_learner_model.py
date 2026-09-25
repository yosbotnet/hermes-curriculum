"""Dialogue mode: the learner model (domain, in-memory adapter, service, MCP routers, CLI render).

Everything runs on the in-memory stack with a FixedClock, so every schedule is
deterministic and no database or API key is needed. The live Postgres round-trip
for the same port is in test_postgres.py (skipped unless a database is configured).
"""
from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta

from curriculum.application.composition import build_in_memory
from curriculum.application.learner_model import REVIEW_HOW_TO, LearnerModelService
from curriculum.application.policies import FixedClock
from curriculum.cli import _build_parser, _render_notes
from curriculum.domain.enums import FsrsRating
from curriculum.domain.learner import OUTCOMES, REVIEWABLE, LearnerNote, NoteKind
from curriculum.engine.fsrs import FsrsScheduler
from curriculum.mcp import server as mcp_server
from curriculum.storage.memory import InMemoryLearnerNoteRepository

NOW = datetime(2026, 9, 25, 12, 0, 0)
COURSE = "GPUKernelOptimization"


class _Clock(FixedClock):
    """A FixedClock that tests can move forward."""

    def advance(self, **kw) -> None:
        self._instant = self._instant + timedelta(**kw)


def _service() -> tuple[LearnerModelService, InMemoryLearnerNoteRepository, _Clock]:
    clock = _Clock(NOW)
    repo = InMemoryLearnerNoteRepository()
    return LearnerModelService(notes=repo, scheduler=FsrsScheduler(), clock=clock), repo, clock


class DomainTests(unittest.TestCase):
    def test_reviewable_kinds_are_insight_and_fixed_misconception(self) -> None:
        self.assertEqual(REVIEWABLE, {NoteKind.INSIGHT, NoteKind.MISCONCEPTION_FIXED})

    def test_resolved_or_non_reviewable_notes_are_not_reviewable(self) -> None:
        base = LearnerNote(id="n", course=COURSE, kind=NoteKind.INSIGHT, text="t", created_at=NOW)
        self.assertTrue(base.reviewable)
        from dataclasses import replace
        self.assertFalse(replace(base, status="resolved").reviewable)
        self.assertFalse(replace(base, kind=NoteKind.OPEN_THREAD).reviewable)

    def test_note_is_frozen(self) -> None:
        note = LearnerNote(id="n", course=COURSE, kind=NoteKind.GOAL, text="t", created_at=NOW)
        with self.assertRaises(FrozenInstanceError):
            note.text = "x"  # type: ignore[misc]

    def test_outcomes_cover_the_four_fsrs_ratings(self) -> None:
        self.assertEqual(sorted(OUTCOMES.values()), sorted(FsrsRating))


class InMemoryRepositoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo = InMemoryLearnerNoteRepository()
        mk = lambda i, kind, at, **kw: LearnerNote(id=i, course=kw.pop("course", COURSE), kind=kind, text=i, created_at=at, **kw)
        for note in (
            mk("b", NoteKind.INSIGHT, NOW, due_at=NOW + timedelta(days=1)),
            mk("a", NoteKind.INSIGHT, NOW, due_at=NOW - timedelta(days=1)),
            mk("c", NoteKind.OPEN_THREAD, NOW - timedelta(hours=1)),
            mk("d", NoteKind.INSIGHT, NOW, due_at=NOW - timedelta(days=2), status="resolved"),
            mk("e", NoteKind.GOAL, NOW, course="PCD"),
        ):
            self.repo.upsert(note)

    def test_list_is_oldest_first_with_id_tiebreak_and_filters(self) -> None:
        self.assertEqual([n.id for n in self.repo.list(COURSE)], ["c", "a", "b", "d"])
        self.assertEqual([n.id for n in self.repo.list(COURSE, kinds=[NoteKind.INSIGHT], status="active")], ["a", "b"])

    def test_due_is_active_reviewable_and_past_due_only(self) -> None:
        self.assertEqual([n.id for n in self.repo.due(COURSE, NOW)], ["a"])

    def test_list_courses(self) -> None:
        self.assertEqual(list(self.repo.list_courses()), ["GPUKernelOptimization", "PCD"])


class RememberTests(unittest.TestCase):
    def setUp(self) -> None:
        self.svc, self.repo, self.clock = _service()

    def test_insight_is_stored_with_learner_words_and_scheduled(self) -> None:
        out = self.svc.remember(kind="insight", course=COURSE, topic="coalescing",
                                text="Separate warp-uniform from varying values",
                                learner_words="fixed vs mobile", source_file="play/gpu/level-1.html", source_line=3)
        self.assertTrue(out["created"])
        note = self.repo.get(out["id"])
        self.assertEqual(note.learner_words, "fixed vs mobile")
        self.assertEqual(note.source_ref.file, "play/gpu/level-1.html")
        self.assertEqual(note.reps, 0)                      # being learned is not a review
        self.assertIsNotNone(note.stability)
        self.assertGreater(out["due_in_days"], 2)           # FSRS cold-start for GOOD: about 3 days
        self.assertLess(out["due_in_days"], 5)

    def test_open_threads_goals_and_gaps_are_not_scheduled(self) -> None:
        for kind in ("open_thread", "goal", "material_gap"):
            out = self.svc.remember(kind=kind, course=COURSE, text=f"a {kind}")
            self.assertNotIn("due_at", out)
            self.assertIsNone(self.repo.get(out["id"]).due_at)

    def test_same_text_is_not_duplicated(self) -> None:
        a = self.svc.remember(kind="insight", course=COURSE, text="Threads side by side matter")
        self.clock.advance(minutes=5)
        b = self.svc.remember(kind="insight", course=COURSE, text="  threads SIDE by side   matter ")
        self.assertFalse(b["created"])
        self.assertEqual(a["id"], b["id"])
        self.assertEqual(len(self.repo.list(COURSE)), 1)

    def test_same_text_different_kind_is_a_new_note(self) -> None:
        a = self.svc.remember(kind="insight", course=COURSE, text="same")
        b = self.svc.remember(kind="open_thread", course=COURSE, text="same")
        self.assertNotEqual(a["id"], b["id"])

    def test_validation(self) -> None:
        with self.assertRaises(ValueError):
            self.svc.remember(kind="fact", course=COURSE, text="x")
        with self.assertRaises(ValueError):
            self.svc.remember(kind="insight", course=COURSE, text="   ")
        with self.assertRaises(ValueError):
            self.svc.remember(kind="insight", course="", text="x")

    def test_flag_material_records_a_gap_at_the_place(self) -> None:
        out = self.svc.flag_material(course=COURSE, source_file="play/gpu/level-1.html",
                                     source_line=40, what_was_unclear="blockIdx used before it was explained")
        self.assertEqual(out["kind"], "material_gap")
        self.assertEqual(out["source_ref"], {"file": "play/gpu/level-1.html", "line": 40})
        self.assertEqual([g["id"] for g in self.svc.gaps(COURSE)], [out["id"]])


class RecallTests(unittest.TestCase):
    def setUp(self) -> None:
        self.svc, self.repo, self.clock = _service()
        self.svc.remember(kind="goal", course=COURSE, text="Write a fast matmul kernel")
        self.svc.remember(kind="insight", course=COURSE, topic="coalescing", text="uniform vs varying")
        self.clock.advance(minutes=1)
        self.svc.remember(kind="insight", course=COURSE, topic="shared memory", text="banks are sectors for smem")
        self.svc.remember(kind="open_thread", course=COURSE, topic="shared memory", text="why padding to 33 works")
        self.svc.remember(kind="misconception_fixed", course=COURSE, topic="warps", text="thought a block was 32 warps")

    def test_everything_newest_first(self) -> None:
        r = self.svc.recall(COURSE)
        self.assertEqual([i["text"] for i in r["insights"]["items"]], ["banks are sectors for smem", "uniform vs varying"])
        self.assertEqual(len(r["goals"]["items"]), 1)
        self.assertEqual(len(r["open_threads"]["items"]), 1)
        self.assertEqual(len(r["misconceptions_fixed"]["items"]), 1)
        self.assertEqual(r["ripe_reviews"], 0)
        self.assertIn("three sentences", r["how_to_use"])

    def test_topic_filters_everything_but_goals(self) -> None:
        r = self.svc.recall(COURSE, topic="shared")
        self.assertEqual([i["text"] for i in r["insights"]["items"]], ["banks are sectors for smem"])
        self.assertEqual(len(r["open_threads"]["items"]), 1)
        self.assertEqual(r["misconceptions_fixed"]["items"], [])
        self.assertEqual(len(r["goals"]["items"]), 1)       # goals always come back

    def test_resolved_threads_disappear_and_keep_their_resolution(self) -> None:
        thread = self.svc.recall(COURSE)["open_threads"]["items"][0]
        done = self.svc.resolve(thread["id"], resolution="row length 33 shifts the bank by one per row")
        self.assertEqual(done["status"], "resolved")
        self.assertIn("-> row length 33", done["text"])
        self.assertEqual(self.svc.recall(COURSE)["open_threads"]["items"], [])

    def test_lists_are_capped_with_a_more_count(self) -> None:
        for i in range(10):
            self.svc.remember(kind="open_thread", course=COURSE, text=f"thread {i}")
        r = self.svc.recall(COURSE)
        self.assertEqual(len(r["open_threads"]["items"]), 8)
        self.assertEqual(r["open_threads"]["more"], 3)

    def test_unknown_note_raises(self) -> None:
        with self.assertRaises(KeyError):
            self.svc.resolve("ln-nope")


class ReviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.svc, self.repo, self.clock = _service()
        self.old = self.svc.remember(kind="insight", course=COURSE, text="uniform vs varying", learner_words="fixed vs mobile")
        self.clock.advance(days=2)
        self.new = self.svc.remember(kind="misconception_fixed", course=COURSE, text="k advances between warps")
        self.thread = self.svc.remember(kind="open_thread", course=COURSE, text="padding")

    def test_nothing_is_ripe_before_its_due_date(self) -> None:
        self.assertEqual(self.svc.reviews(COURSE)["ripe_total"], 0)

    def test_ripe_notes_come_most_at_risk_first_with_instructions(self) -> None:
        self.clock.advance(days=10)
        r = self.svc.reviews(COURSE)
        self.assertEqual(r["ripe_total"], 2)
        self.assertEqual([x["id"] for x in r["reviews"]], [self.old["id"], self.new["id"]])  # older = lower recall
        self.assertLess(r["reviews"][0]["recall_probability"], r["reviews"][1]["recall_probability"])
        self.assertEqual(r["reviews"][0]["learner_words"], "fixed vs mobile")
        self.assertEqual(r["how_to_use"], REVIEW_HOW_TO)
        self.assertIn("new case", r["how_to_use"])
        self.assertEqual(len(self.svc.reviews(COURSE, limit=1)["reviews"]), 1)

    def test_applied_pushes_the_next_review_further_than_forgot(self) -> None:
        self.clock.advance(days=10)
        good = self.svc.review_result(self.old["id"], "applied")
        bad = self.svc.review_result(self.new["id"], "forgot")
        self.assertGreater(good["interval_days"], bad["interval_days"])
        self.assertEqual(good["reviews_done"], 1)
        self.assertEqual(self.repo.get(self.new["id"]).lapses, 1)
        self.assertEqual(self.svc.reviews(COURSE)["ripe_total"], 0)

    def test_outcome_aliases(self) -> None:
        self.clock.advance(days=10)
        self.assertEqual(self.svc.review_result(self.old["id"], "GOOD")["outcome"], "applied")
        self.assertEqual(self.svc.review_result(self.new["id"], "2")["outcome"], "struggled")

    def test_errors(self) -> None:
        with self.assertRaises(ValueError):
            self.svc.review_result(self.old["id"], "meh")
        with self.assertRaises(ValueError):
            self.svc.review_result(self.thread["id"], "applied")    # open threads are not reviewed
        with self.assertRaises(KeyError):
            self.svc.review_result("ln-nope", "applied")


class WiringTests(unittest.TestCase):
    def test_in_memory_stack_carries_the_learner_service(self) -> None:
        stack = build_in_memory(clock=FixedClock(NOW))
        stack.learner.remember(kind="goal", course=COURSE, text="PCD oral")
        self.assertEqual(stack.learner.list_courses(), [COURSE])
        self.assertEqual(len(stack.notes.list(COURSE)), 1)

    def test_routers_return_plain_dicts(self) -> None:
        stack = build_in_memory(clock=FixedClock(NOW))
        learner = stack.learner
        out = mcp_server._call_remember(learner, kind="open_thread", course=COURSE, text="why 33")
        self.assertIsInstance(out, dict)
        self.assertIsInstance(mcp_server._call_recall(learner, COURSE), dict)
        self.assertIsInstance(mcp_server._call_reviews(learner, COURSE), dict)
        self.assertEqual(mcp_server._call_resolve(learner, out["id"])["status"], "resolved")
        gap = mcp_server._call_flag_material(learner, course=COURSE, source_file="f.html", what_was_unclear="x")
        self.assertEqual(gap["kind"], "material_gap")

    def test_tool_names_are_stable(self) -> None:
        self.assertEqual(mcp_server.LEARNER_TOOL_NAMES,
                         ("recall", "remember", "resolve", "reviews", "review_result", "flag_material"))
        self.assertTrue(set(mcp_server.LEARNER_TOOL_NAMES).isdisjoint(mcp_server.TOOL_NAMES))

    @unittest.skipUnless(mcp_server.FastMCP is not None, "needs the mcp package")
    def test_server_registers_learner_tools_only_when_wired(self) -> None:
        import asyncio

        stack = build_in_memory(clock=FixedClock(NOW))
        with_learner = {t.name for t in asyncio.run(mcp_server.build_server(stack.service, stack.learner).list_tools())}
        without = {t.name for t in asyncio.run(mcp_server.build_server(stack.service).list_tools())}
        self.assertEqual(with_learner, set(mcp_server.TOOL_NAMES) | set(mcp_server.LEARNER_TOOL_NAMES))
        self.assertEqual(without, set(mcp_server.TOOL_NAMES))


class CliTests(unittest.TestCase):
    def test_new_commands_parse(self) -> None:
        parser = _build_parser()
        a = parser.parse_args(["remember", "insight", "uniform vs varying", "--course", COURSE,
                               "--words", "fixed vs mobile", "--source", "play/gpu/level-1.html:12"])
        self.assertEqual((a.kind, a.words, a.source), ("insight", "fixed vs mobile", "play/gpu/level-1.html:12"))
        self.assertTrue(parser.parse_args(["notes", "--all"]).all)
        self.assertIsNotNone(parser.parse_args(["db-migrate"]).func)

    def test_render_groups_by_kind_and_shows_learner_words(self) -> None:
        stack = build_in_memory(clock=FixedClock(NOW))
        stack.learner.remember(kind="insight", course=COURSE, topic="coalescing", text="uniform vs varying", learner_words="fixed vs mobile")
        stack.learner.remember(kind="goal", course=COURSE, text="fast matmul")
        text = _render_notes(list(stack.learner.notes(COURSE)))
        self.assertLess(text.index("goal (1)"), text.index("insight (1)"))
        self.assertIn('"fixed vs mobile"', text)
        self.assertIn("[coalescing]", text)
        self.assertIn("review in", text)
        self.assertEqual(_render_notes([]), "(no notes)")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

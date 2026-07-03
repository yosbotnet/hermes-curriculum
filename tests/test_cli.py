"""Tests for the agent-facing CLI (curriculum.cli).

Stdlib unittest only and strictly offline: the contract these tests pin is the
*shell* of the CLI -- argv parsing, exit codes, and that the dependency-free
commands (``--help`` and ``doctor``) work on a fresh machine -- NOT the build
orchestration, whose collaborators own those tests. So we never call a command
that would touch the network or the database: the only data-driven command we
exercise is ``doctor``, with its database probe monkeypatched to a canned result
so no socket is ever opened.

The load-bearing properties asserted here:

* ``main`` returns an ``int`` and never raises, even on argparse's own exit paths
  (``--help`` -> 0, an unknown subcommand -> non-zero), because it converts
  ``SystemExit`` into a return code.
* a bare invocation prints usage instead of crashing.
* ``doctor`` runs end to end (every probe isolated) and reports a checklist.
* ``check`` and ``flag-question`` drive the service through the composition
  root, which is monkeypatched to a fake in-memory service (same isolation
  principle as the ``doctor`` database probe) so no Postgres connection or
  OKF bundle is ever touched.
"""
from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from curriculum import cli


class _FakeService:
    """A minimal stand-in for CurriculumApplicationService: only the two
    methods the CLI's motivation-layer commands call, recording their
    arguments so tests can assert on what the CLI passed through."""

    def __init__(self, checkin_payload=None, flag_result=None):
        self.checkin_payload = checkin_payload
        self.flag_result = flag_result
        self.checkin_calls: list[str] = []
        self.flag_calls: list[tuple[str, str]] = []

    def checkin(self, course):
        self.checkin_calls.append(course)
        return self.checkin_payload

    def flag_question(self, question_id, *, reason=""):
        self.flag_calls.append((question_id, reason))
        return self.flag_result


_CHECK_PAYLOAD = {
    "course": "cyber-101",
    "stability_days": 42.7,
    "delta_since_last_check": 3.2,
    "consolidation": {"holding": 12, "reviewed_since": 4},
    "ripeness": {
        "ready_now": ["c1", "c2"],
        "ready_tomorrow": ["c3"],
        "ready_this_week": ["c4", "c5", "c6"],
        "holding": ["c7"],
    },
    "unlocks_ready": ["c8"],
    "near_unlocks": [{"concept_id": "c9", "missing": 1, "one_away": True}],
    "by_mastery": {"new": 2, "learning": 3, "solid": 4, "exam_ready": 1},
}


class HelpAndUsageTest(unittest.TestCase):
    def test_help_returns_int_and_prints_usage(self) -> None:
        # --help makes argparse print to stdout and raise SystemExit(0); main must
        # absorb that and hand back a plain int.
        out = io.StringIO()
        with redirect_stdout(out):
            code = cli.main(["--help"])
        self.assertIsInstance(code, int)
        self.assertEqual(code, 0)
        self.assertIn("usage", out.getvalue().lower())

    def test_no_args_prints_usage_and_returns_int(self) -> None:
        out = io.StringIO()
        with redirect_stdout(out):
            code = cli.main([])
        self.assertIsInstance(code, int)
        self.assertIn("usage", out.getvalue().lower())

    def test_unknown_subcommand_returns_nonzero(self) -> None:
        # An invalid choice makes argparse print an error to stderr and exit 2;
        # the caller should see a non-zero int, not an exception.
        err = io.StringIO()
        with redirect_stderr(err):
            code = cli.main(["definitely-not-a-command"])
        self.assertIsInstance(code, int)
        self.assertNotEqual(code, 0)


class DoctorTest(unittest.TestCase):
    def test_doctor_runs_without_raising(self) -> None:
        # Pin the DB probe so the test opens no socket (honouring the no-network /
        # no-DB rule) while the docker/key/bundle probes run for real -- they only
        # consult PATH, settings, and the filesystem, so they are side-effect free.
        out = io.StringIO()
        with mock.patch.object(
            cli, "_check_db", return_value=("database", False, "skipped in test")
        ):
            with redirect_stdout(out):
                code = cli.main(["doctor"])
        self.assertIsInstance(code, int)
        printed = out.getvalue().lower()
        # Every probe is reported, each line marked OK or MISS.
        self.assertIn("docker", printed)
        self.assertIn("database", printed)
        self.assertIn("nous_api_key", printed)
        self.assertTrue("ok" in printed or "miss" in printed)

    def test_doctor_returns_nonzero_when_a_check_misses(self) -> None:
        # With the DB probe forced to miss, doctor must report a non-zero status
        # so it can double as a scriptable readiness gate.
        with mock.patch.object(
            cli, "_check_db", return_value=("database", False, "skipped in test")
        ):
            with redirect_stdout(io.StringIO()):
                code = cli.main(["doctor"])
        self.assertNotEqual(code, 0)

    def test_doctor_reports_log_dir(self) -> None:
        # doctor gains one line for the build-log directory and whether it is
        # writable, following the existing OK/MISS probe style.
        out = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            env = {"CURRICULUM_LOG_DIR": directory}
            with mock.patch.dict(os.environ, env, clear=False):
                with mock.patch.object(
                    cli, "_check_db", return_value=("database", False, "skipped in test")
                ):
                    with redirect_stdout(out):
                        cli.main(["doctor"])
        printed = out.getvalue().lower()
        self.assertIn("log dir", printed)


class BuildLogPathTest(unittest.TestCase):
    """The build/ingest/link/questions commands print the durable log path to
    stdout at the START of the run so the operator can find it later even if the
    run is killed."""

    def test_ingest_prints_log_path_to_stdout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "corpus.json"
            manifest_path.write_text(
                json.dumps(
                    {"course": "C", "sources": [{"path": "a.txt", "token": "a"}]}
                ),
                encoding="utf-8",
            )
            out = io.StringIO()
            env = {"CURRICULUM_LOG_DIR": directory}
            with mock.patch.dict(os.environ, env, clear=False):
                # Stub the orchestration so the test stays offline: we only assert
                # the CLI's log-path plumbing, not a real ingest.
                with mock.patch(
                    "curriculum.app.build.ingest", return_value={"files": 0}
                ):
                    with redirect_stdout(out):
                        code = cli.main(["ingest", str(manifest_path)])
            self.assertEqual(code, 0)
            printed = out.getvalue()
            self.assertIn("build-", printed)  # the log filename prefix
            self.assertIn(directory, printed)  # under the overridden log dir
            # A log file was actually created under the override dir.
            logs = list(Path(directory).glob("build-*.log"))
            self.assertTrue(logs, "expected a build log file to be created")


class CheckTest(unittest.TestCase):
    def test_check_prints_gain_framed_report_and_returns_zero(self) -> None:
        fake = _FakeService(checkin_payload=_CHECK_PAYLOAD)
        out = io.StringIO()
        with mock.patch(
            "curriculum.application.composition.build_service", return_value=fake
        ):
            with redirect_stdout(out):
                code = cli.main(["check", "--course", "cyber-101"])
        self.assertEqual(code, 0)
        self.assertEqual(fake.checkin_calls, ["cyber-101"])
        printed = out.getvalue()
        lines = printed.splitlines()
        self.assertTrue(any(line.startswith("Knowledge held:") for line in lines))
        self.assertTrue(any(line.startswith("Ready today:") for line in lines))
        self.assertTrue(any(line.startswith("Unlocked:") for line in lines))
        # Gain-framed: never render this as a list of obligations.
        lowered = printed.lower()
        for banned in ("overdue", "debt", "behind", "late"):
            self.assertNotIn(banned, lowered)

    def test_check_json_emits_raw_payload(self) -> None:
        fake = _FakeService(checkin_payload=_CHECK_PAYLOAD)
        out = io.StringIO()
        with mock.patch(
            "curriculum.application.composition.build_service", return_value=fake
        ):
            with redirect_stdout(out):
                code = cli.main(["check", "--course", "cyber-101", "--json"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out.getvalue()), _CHECK_PAYLOAD)


class RenderCheckTest(unittest.TestCase):
    """_render_check is a small pure function: exercise it directly, no CLI,
    no service, so its output contract is pinned independent of wiring."""

    def test_render_check_includes_required_lines(self) -> None:
        rendered = cli._render_check(_CHECK_PAYLOAD)
        lines = rendered.splitlines()
        self.assertLessEqual(len(lines), 12)
        self.assertTrue(any(line.startswith("Knowledge held:") for line in lines))
        self.assertTrue(any(line.startswith("Ready today:") for line in lines))
        self.assertTrue(any(line.startswith("Unlocked:") for line in lines))
        lowered = rendered.lower()
        for banned in ("overdue", "debt", "behind", "late"):
            self.assertNotIn(banned, lowered)
        # Plain ASCII only.
        rendered.encode("ascii")

    def test_render_check_handles_first_check_with_no_delta(self) -> None:
        payload = dict(_CHECK_PAYLOAD, delta_since_last_check=None)
        rendered = cli._render_check(payload)
        self.assertIn("first check", rendered)
        # The unconditional lines must hold on this branch too.
        lines = rendered.splitlines()
        self.assertTrue(any(line.startswith("Knowledge held:") for line in lines))
        self.assertTrue(any(line.startswith("Ready today:") for line in lines))
        self.assertTrue(any(line.startswith("Unlocked:") for line in lines))

    def test_render_check_omits_verge_line_without_one_away(self) -> None:
        # No one_away rows -> the "On the verge" line must not appear at all.
        payload = dict(
            _CHECK_PAYLOAD,
            near_unlocks=[{"concept_id": "c9", "missing": 2, "one_away": False}],
        )
        rendered = cli._render_check(payload)
        self.assertNotIn("On the verge", rendered)
        payload = dict(_CHECK_PAYLOAD, near_unlocks=[])
        rendered = cli._render_check(payload)
        self.assertNotIn("On the verge", rendered)

    def test_render_check_formats_negative_delta_and_zero_counts(self) -> None:
        payload = dict(
            _CHECK_PAYLOAD,
            delta_since_last_check=-3.2,
            ripeness={
                "ready_now": [],
                "ready_tomorrow": [],
                "ready_this_week": [],
                "holding": [],
            },
            unlocks_ready=[],
        )
        rendered = cli._render_check(payload)
        self.assertIn("(-3.2 since last check)", rendered)
        self.assertIn("Ready today: 0", rendered)
        self.assertIn("Unlocked: 0 new concept(s)", rendered)


class FlagQuestionTest(unittest.TestCase):
    def test_flag_question_calls_service_and_returns_zero(self) -> None:
        fake = _FakeService(flag_result={"question_id": "q1", "status": "retired"})
        out = io.StringIO()
        with mock.patch(
            "curriculum.application.composition.build_service", return_value=fake
        ):
            with redirect_stdout(out):
                code = cli.main(["flag-question", "q1", "--reason", "ambiguous"])
        self.assertEqual(code, 0)
        self.assertEqual(fake.flag_calls, [("q1", "ambiguous")])
        self.assertEqual(
            json.loads(out.getvalue()), {"question_id": "q1", "status": "retired"}
        )


if __name__ == "__main__":
    unittest.main()

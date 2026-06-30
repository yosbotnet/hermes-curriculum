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
"""
from __future__ import annotations

import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

from curriculum import cli


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


if __name__ == "__main__":
    unittest.main()

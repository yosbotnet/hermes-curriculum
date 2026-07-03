"""Tests for durable per-invocation build logging (curriculum.app.build_logging).

Stdlib unittest only and strictly offline. These pin the whole point of issue #3:
a build that stalls, times out, or is killed must still leave a durable, findable
log artifact on disk. So the assertions cover the predictable filename (UTC
timestamp + pid), that the file exists and is flushed per record (not buffered in
memory until the end), that the ``CURRICULUM_LOG_DIR`` override is honoured, and
that a FAILING ingest leaves the stage name and the FULL provider/error traceback
in the file. No test touches the network or the database: the ingest failure is
forced by pointing a source at a path that does not exist, which raises before any
Nous call, and every other collaborator is a pure helper.
"""
from __future__ import annotations

import logging
import os
import re
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from curriculum.app import build, build_logging
from curriculum.config import Settings, load as load_settings

# build-YYYYMMDDThhmmssZ-<pid>.log
_NAME_RE = re.compile(r"^build-\d{8}T\d{6}Z-\d+\.log$")


class LogPathTest(unittest.TestCase):
    def test_filename_carries_utc_timestamp_and_pid(self) -> None:
        settings = Settings(log_dir="logs")
        moment = datetime(2026, 7, 3, 14, 20, 0, tzinfo=timezone.utc)
        path = build_logging.log_path_for(settings, "build", now=moment, pid=12345)
        self.assertEqual(path.name, "build-20260703T142000Z-12345.log")
        self.assertTrue(_NAME_RE.match(path.name), path.name)
        self.assertEqual(path.parent, Path("logs"))

    def test_default_now_and_pid_still_match_the_pattern(self) -> None:
        settings = Settings(log_dir="logs")
        path = build_logging.log_path_for(settings, "ingest")
        self.assertTrue(_NAME_RE.match(path.name), path.name)


class StartBuildLogTest(unittest.TestCase):
    def test_creates_file_at_predictable_path_and_flushes_header(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(log_dir=directory)
            logger, path = build_logging.start_build_log(settings, "build")
            try:
                # Durable: the header must be on disk BEFORE any close/flush at the
                # end -- read it back mid-run to prove nothing is buffered in memory.
                self.assertTrue(path.exists())
                self.assertTrue(_NAME_RE.match(path.name), path.name)
                self.assertEqual(path.parent, Path(directory))
                logger.info("stage=ingest source ok token=alpha")
                text = path.read_text(encoding="utf-8")
                self.assertIn("build", text)  # command in the header
                self.assertIn("stage=ingest", text)
                self.assertIn(str(os.getpid()), text)  # pid on the line
            finally:
                build_logging.close_build_log(logger)

    def test_timestamps_are_utc_iso(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(log_dir=directory)
            logger, path = build_logging.start_build_log(settings, "link")
            try:
                logger.info("hello")
            finally:
                build_logging.close_build_log(logger)
            text = path.read_text(encoding="utf-8")
            # UTC ISO-ish, trailing Z: e.g. 2026-07-03T14:20:00Z
            self.assertRegex(text, r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")

    def test_close_is_idempotent_and_detaches_handlers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(log_dir=directory)
            logger, _ = build_logging.start_build_log(settings, "build")
            build_logging.close_build_log(logger)
            build_logging.close_build_log(logger)  # must not raise
            self.assertEqual(logger.handlers, [])


class LogDirOverrideTest(unittest.TestCase):
    def test_env_override_is_respected(self) -> None:
        settings = load_settings({"CURRICULUM_LOG_DIR": "/tmp/custom-logs"})
        self.assertEqual(settings.log_dir, "/tmp/custom-logs")

    def test_default_log_dir_is_logs(self) -> None:
        self.assertEqual(load_settings({}).log_dir, "logs")


class IngestFailureLoggingTest(unittest.TestCase):
    """A failing ingest must leave the partial log on disk with the stage name and
    the FULL error traceback -- that is the entire point of the issue."""

    def test_failing_source_records_stage_and_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(
                log_dir=directory,
                nous_api_key="test-key",  # present so ingest gets past the key gate
                okf_bundle_path=str(Path(directory) / "bundle"),
            )
            logger, path = build_logging.start_build_log(settings, "ingest")
            try:
                manifest = {
                    "course": "C",
                    "chunk_lines": 150,
                    # This path does not exist: reading it raises inside the worker
                    # BEFORE any Nous call, so the failure is offline + deterministic.
                    "sources": [{"path": str(Path(directory) / "missing.txt"), "token": "src-a"}],
                }
                result = build.ingest(manifest, settings, logger=logger)
            finally:
                build_logging.close_build_log(logger)

            # One bad source is tolerated (the batch is not aborted); it just does
            # not count toward files.
            self.assertEqual(result["files"], 0)
            text = path.read_text(encoding="utf-8")
            self.assertIn("stage=ingest", text)
            self.assertIn("src-a", text)  # per-source identity preserved
            self.assertIn("Traceback (most recent call last)", text)  # FULL traceback


if __name__ == "__main__":
    unittest.main()

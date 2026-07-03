"""Durable per-invocation build logging (app layer; stdlib ``logging`` only).

Issue #3: when ``curriculum build`` (or a single ingest / link / questions stage)
stalls, times out, or is killed, its diagnostics used to die with the terminal
session. This module gives every build invocation ONE durable log file so a
post-mortem is always possible.

Design contract (why it is shaped this way)
--------------------------------------------
* **App layer, not engine.** The engine (``engine``/``domain``/``ports``) stays
  pure -- it never logs. Only this module and the CLI handlers that call it own
  logging, so the pure core keeps its no-I/O guarantee.
* **Predictable path.** One file per invocation under ``settings.log_dir`` (the
  ``CURRICULUM_LOG_DIR`` setting, default ``logs``), named with a UTC timestamp
  and the process id -- ``build-20260703T142000Z-12345.log``. Timestamp + pid
  make the name unique and sortable, and let a killed run be found later.
* **Durable under a kill.** The handler is an ordinary
  :class:`logging.FileHandler`, whose ``emit`` flushes after every record (it is
  a ``StreamHandler``), so nothing is buffered in memory until the end: a process
  killed mid-run still leaves the partial log on disk, which is the entire point.
* **UTC ISO timestamps + pid on every line.** The formatter stamps each record
  with a UTC ISO instant and the pid, so interleaved stages remain attributable.

Standard library only.
"""
from __future__ import annotations

import itertools
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from ..config import Settings

__all__ = ["log_path_for", "start_build_log", "close_build_log"]

_LOGGER_NAMESPACE = "curriculum.build"
# Per-process monotonic counter so two builds started in the same second (same
# pid) still get distinct logger names -- otherwise ``logging.getLogger`` would
# hand back the same singleton and stack a second FileHandler on it.
_SEQUENCE = itertools.count()

# A shared no-op logger for orchestration code called WITHOUT a real build log
# (e.g. a unit test, or an embedded caller): logging becomes a cheap no-op rather
# than requiring every call site to branch on ``logger is None``.
NULL_LOGGER = logging.getLogger(f"{_LOGGER_NAMESPACE}.null")
NULL_LOGGER.addHandler(logging.NullHandler())
NULL_LOGGER.propagate = False


def log_path_for(
    settings: Settings,
    command: str,
    *,
    now: datetime | None = None,
    pid: int | None = None,
) -> Path:
    """Return the predictable log-file path for one build invocation.

    The name is always ``build-<UTC-timestamp>-<pid>.log`` regardless of
    ``command`` (the command is recorded INSIDE the file), so every invocation's
    artifact sorts chronologically in the log directory. ``now``/``pid`` are
    injectable purely so the naming can be asserted deterministically in a test.
    """
    moment = now or datetime.now(timezone.utc)
    process_id = os.getpid() if pid is None else pid
    stamp = moment.strftime("%Y%m%dT%H%M%SZ")
    return Path(settings.log_dir) / f"build-{stamp}-{process_id}.log"


def start_build_log(settings: Settings, command: str) -> tuple[logging.Logger, Path]:
    """Open this invocation's durable log; return ``(logger, path)``.

    Creates ``settings.log_dir`` if needed, attaches a per-record-flushing
    :class:`logging.FileHandler`, writes a header line (command + pid), and hands
    back both the logger to thread through the build and the path to print to the
    operator up front. The logger does not propagate to the root logger, so build
    lines never leak into whatever handlers the host process installed.
    """
    path = log_path_for(settings, command)
    path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"{_LOGGER_NAMESPACE}.{os.getpid()}.{next(_SEQUENCE)}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    # Fresh start: this name is unique, but clear defensively so a reused logger
    # never accumulates duplicate handlers writing the same line twice.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    # mode="w": one file per invocation. delay=False: open now so the file exists
    # (and the header lands) the instant the build starts, even before any stage.
    handler = logging.FileHandler(path, mode="w", encoding="utf-8", delay=False)
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s pid=%(process)d %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    formatter.converter = time.gmtime  # UTC, not local time
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info("build log opened command=%s pid=%s path=%s", command, os.getpid(), path)
    return logger, path


def close_build_log(logger: logging.Logger) -> None:
    """Flush, close, and detach this invocation's handlers (idempotent).

    Safe to call twice and safe to call on the shared :data:`NULL_LOGGER` (whose
    only handler is a no-op), so a ``finally`` block can always close without
    guarding. Detaching releases the file handle so tests can clean up their temp
    directories on every platform.
    """
    for handler in list(logger.handlers):
        try:
            handler.flush()
            handler.close()
        finally:
            logger.removeHandler(handler)

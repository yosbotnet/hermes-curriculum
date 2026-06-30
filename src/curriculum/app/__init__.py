"""Config-driven build orchestration (the reproducible, agent-bootstrappable seam).

This package turns the throwaway one-off ingest scripts into a small, importable
API that drives the whole corpus build from a *manifest* plus a
:class:`curriculum.config.Settings` -- no hardcoded paths, no inlined API keys,
no per-user filesystem assumptions. The CLI and the test-suite both call into
:mod:`curriculum.app.build`, so a fresh checkout (or an agent) can reproduce a
course graph end to end from declared inputs alone.

The functions are re-exported here for convenience so callers can write
``from curriculum.app import build, ingest`` without reaching into the submodule.
Importing this package pulls in :mod:`curriculum.app.build`, which is itself
import-light: the heavyweight, optional collaborators (psycopg and the embedding
linker) are imported lazily *inside* the functions, so ``import curriculum.app``
succeeds on a machine with neither a database driver nor a network.
"""
from __future__ import annotations

from .build import (
    generate_questions,
    ingest,
    link,
    load_manifest,
    status,
)

__all__ = [
    "load_manifest",
    "ingest",
    "link",
    "generate_questions",
    "status",
]

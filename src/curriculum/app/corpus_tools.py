"""Corpus-preparation onboarding: bridge raw materials to a buildable corpus.

New users arrive with PDFs, slide decks, and lecture notes, but the build
pipeline (:mod:`curriculum.app.build`) ingests only plain UTF-8 text listed in a
``corpus.json`` manifest. Nothing in the repo used to bridge that gap, so people
stalled at the very first step. This module is that bridge, and it is shaped for
the way the repo is actually operated -- through a coding agent:

* :func:`scaffold` lays down the starting point (a ``materials/`` directory, a
  commented ``corpus.json`` template, and a ``materials/README.txt``) so the agent
  has a concrete place to write the text it extracts. It REFUSES to overwrite an
  existing manifest, because re-running the scaffold must never destroy a corpus a
  user has already begun filling in.
* :func:`validate` is a machine-checkable report the agent can loop against: it
  names the concrete ways each source is not yet ingestable (missing file, a
  still-binary PDF/zip, non-UTF-8, empty, suspiciously short), warns when no
  trusted spine ordering is declared, and estimates the extract-call cost BEFORE
  any paid inference runs -- so cost is visible up front and the agent iterates on
  ``validate`` until it is clean, then runs ``build``.

Pure standard library: no network, no database. :func:`validate` reuses
:func:`curriculum.app.build.load_manifest` for shape validation (folding its
:class:`ConfigError` into the report rather than raising) and then only reads the
listed files off disk. The companion ``AGENTS.md`` "Preparing a corpus from raw
materials" section is the human-facing runbook these two functions serve.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..domain.errors import ConfigError
from .build import load_manifest

__all__ = ["scaffold", "validate"]

# A source under this many decodable characters is almost certainly a stub (an
# empty extract, a single heading) rather than real teaching text, so it earns a
# warning -- not an error, since a deliberately tiny satellite is legal.
_MIN_CHARS = 200

# How many bytes we sniff to classify a file as binary. A PDF/zip magic number
# lives in the first four bytes; a stray null byte anywhere in this window is the
# tell-tale of an undecoded binary, so an 8 KB window catches it cheaply without
# reading a large file in full.
_SNIFF_BYTES = 8192

# The commented starter manifest. Mirrors ``corpus.example.json`` (``_comment``
# keys, one spine source + satellites) but points at ``materials/`` paths that do
# not exist yet, so the agent's job is simply to drop the extracted text there.
_TEMPLATE: dict[str, Any] = {
    "_comment": (
        "Starter manifest written by 'curriculum corpus-init'. Put plain-text "
        "extracts of YOUR materials under materials/, point each source 'path' at "
        "one, then loop 'curriculum corpus-validate corpus.json' until zero errors "
        "and run 'curriculum build corpus.json'. See the AGENTS.md corpus prep "
        "section. 'token' is a short stable id used as the grounding citation."
    ),
    "course": "YourCourse",
    "chunk_lines": 150,
    "sources": [
        {
            "_comment": (
                "spine:true marks the editorially ordered backbone (e.g. a "
                "textbook's chapter sequence). Its concepts chain into trusted "
                "prerequisite edges in document order; a multi-file spine is "
                "chained across files in manifest order, and a single concatenated "
                "file works too."
            ),
            "path": "materials/textbook.txt",
            "token": "textbook",
            "spine": True,
        },
        {"path": "materials/lecture-01.txt", "token": "lecture-01"},
    ],
}

_README = """\
Put plain-text extracts of your course materials in this directory.

- One file per spine chapter, OR one concatenated file for the whole spine --
  both work; a multi-file spine is chained in manifest order.
- Put each satellite (lecture, tutorial, notes) in its own separate file.
- Extract text first: these must be plain UTF-8 text, not PDF/docx/pptx. Strip
  page furniture (running headers/footers, page numbers) before saving.
- Point each source 'path' in ../corpus.json at one of these files.
See the AGENTS.md "Preparing a corpus from raw materials" section for the runbook.
"""


# --------------------------------------------------------------------------- #
# Scaffold: lay down the starting artifacts (never clobber an existing corpus).
# --------------------------------------------------------------------------- #
def scaffold(directory: str) -> list[str]:
    """Create the starter corpus artifacts under ``directory``; return their paths.

    Writes three things: ``materials/`` (where the agent drops extracted text),
    a commented ``corpus.json`` template (one ``spine`` source plus one satellite,
    both pointing at ``materials/`` paths that do not exist yet), and
    ``materials/README.txt`` (a short reminder of the plain-text convention).

    REFUSES to overwrite an existing ``corpus.json`` -- it raises
    :class:`ConfigError` and touches nothing rather than clobbering a manifest a
    user has already started editing. The ``materials/`` dir and its README are
    created idempotently (``mkdir(exist_ok=True)``), so re-running after only the
    manifest guard has been cleared is safe.

    Returns the list of created/ensured paths (as strings) so the CLI can echo
    exactly what landed on disk.
    """
    root = Path(directory)
    manifest_path = root / "corpus.json"
    if manifest_path.exists():
        raise ConfigError(
            f"refusing to overwrite existing {manifest_path}; delete it first or "
            "scaffold into a fresh directory"
        )
    materials = root / "materials"
    materials.mkdir(parents=True, exist_ok=True)
    readme_path = materials / "README.txt"

    manifest_path.write_text(
        json.dumps(_TEMPLATE, indent=2) + "\n", encoding="utf-8"
    )
    readme_path.write_text(_README, encoding="utf-8")
    return [str(manifest_path), str(materials), str(readme_path)]


# --------------------------------------------------------------------------- #
# Validate: a machine-checkable readiness report the agent loops against.
# --------------------------------------------------------------------------- #
def validate(manifest_path: str) -> dict:
    """Check a manifest and its source files, returning a structured report.

    The report is ``{"errors": [...], "warnings": [...], "sources": [...],
    "estimate": {...}}`` -- an agent loops on it until ``errors`` is empty, weighs
    the warnings, then runs the build.

    Manifest SHAPE is validated by reusing :func:`curriculum.app.build.load_manifest`;
    a :class:`ConfigError` from it is folded into ``errors`` and returned
    immediately (there are no sources to check if the manifest itself is broken).

    Each source is then read and classified (order matters -- the first failing
    check wins, so a binary file is reported as binary, not as "non-UTF-8"):

    * missing file -> error;
    * binary sniff on the first bytes -> error with a pointed hint: a ``%PDF-``
      magic says extract the PDF's text first, a ``PK\\x03\\x04`` (zip container:
      docx/epub/pptx) says extract it first, a null byte says generic binary;
    * not decodable as UTF-8 -> error;
    * empty (zero decoded characters) -> error;
    * under ~200 characters -> warning (suspiciously short, likely a stub).

    Every source that is not a hard ``error`` (an ``"ok"`` source and a short-but-
    valid ``"warning"`` source alike) will be ingested, so it contributes its
    line/chunk counts to the estimate. Spine consideration: a ``spine`` source
    counts as a usable ordering unless it is itself in ``error``; only when NO such
    spine remains does a warning note that every prerequisite edge will be inferred
    and suggest marking a trusted ordering.

    ``estimate`` totals the extract-call cost so it is visible before any paid
    inference: per-source line count and chunk count at the manifest's
    ``chunk_lines``, plus the total chunk/extract-call counts (one extract call per
    chunk).
    """
    report: dict[str, Any] = {
        "errors": [],
        "warnings": [],
        "sources": [],
        "estimate": {},
    }

    try:
        manifest = load_manifest(manifest_path)
    except ConfigError as exc:
        # A malformed manifest has no sources to inspect: report and return.
        report["errors"].append(str(exc))
        return report

    chunk_lines = manifest["chunk_lines"]
    total_chunks = 0
    any_clean_spine = False

    for source in manifest["sources"]:
        row = _check_source(source, chunk_lines)
        report["sources"].append(row)
        # Attribute the source's error/warning to the top-level lists, tagged with
        # its token so the agent knows which file to fix.
        if row["error"]:
            report["errors"].append(f"[{source['token']}] {row['error']}")
        if row["warning"]:
            report["warnings"].append(f"[{source['token']}] {row['warning']}")
        # Anything that is not a hard error WILL be ingested (an "ok" source and a
        # short-but-valid "warning" source alike), so its chunks count toward the
        # cost estimate and, when it is a spine, it supplies a usable ordering.
        # Gating these on status == "ok" undercounts cost and spuriously warns
        # "no spine" for a spine whose only defect is the short-file warning.
        if row["status"] != "error":
            total_chunks += row["chunks"]
            if source.get("spine"):
                any_clean_spine = True

    if not any_clean_spine:
        report["warnings"].append(
            "no spine source: all prerequisite edges will be inferred; consider "
            "marking a trusted ordering with \"spine\": true on the source whose "
            "sequence is editorially vetted"
        )

    report["estimate"] = {
        "chunks": total_chunks,
        "extract_calls": total_chunks,  # ingest issues one extract call per chunk
        "chunk_lines": chunk_lines,
    }
    return report


def _check_source(source: dict, chunk_lines: int) -> dict:
    """Classify one source file, returning its per-source report row.

    The row is ``{token, path, status, error, warning, lines, chunks}``: ``status``
    is ``"ok"``, ``"error"``, or ``"warning"``; ``error``/``warning`` hold the
    single most pointed message (or ``""``); ``lines``/``chunks`` are populated
    only once the file decodes (0 otherwise). ``source['path']`` is trusted as-is:
    :func:`curriculum.app.build.load_manifest` has already resolved it to an
    absolute path against the manifest's directory, so validate reads exactly the
    file the build will -- there is no CWD-relative re-resolution here.
    """
    token = source["token"]
    raw_path = source["path"]
    path = Path(raw_path)

    row: dict[str, Any] = {
        "token": token,
        "path": raw_path,
        "status": "ok",
        "error": "",
        "warning": "",
        "lines": 0,
        "chunks": 0,
    }

    if not path.is_file():
        row["status"] = "error"
        row["error"] = f"missing file: {raw_path} does not exist"
        return row

    data = path.read_bytes()
    binary_hint = _binary_hint(data)
    if binary_hint is not None:
        row["status"] = "error"
        row["error"] = binary_hint
        return row

    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        row["status"] = "error"
        row["error"] = (
            f"not valid UTF-8 (byte {exc.start}): re-extract or re-encode as "
            "UTF-8 plain text"
        )
        return row

    if len(text) == 0:
        row["status"] = "error"
        row["error"] = "empty file: no text to ingest"
        return row

    # Decoded cleanly: record the extraction cost this source contributes.
    lines = text.splitlines()
    row["lines"] = len(lines)
    row["chunks"] = _chunk_count(lines, chunk_lines)

    if len(text) < _MIN_CHARS:
        row["status"] = "warning"
        row["warning"] = (
            f"suspiciously short ({len(text)} chars < {_MIN_CHARS}); is this the "
            "full extract?"
        )
    return row


def _binary_hint(data: bytes) -> str | None:
    """Return a pointed 'this is binary' message, or ``None`` if it looks like text.

    Checks the most specific containers first so the message can name the actual
    format and tell the agent what to do: a ``%PDF-`` magic (extract the PDF's
    text), a ``PK\\x03\\x04`` zip magic (docx/epub/pptx container -- extract it),
    then a generic null byte anywhere in the sniff window (undecoded binary of some
    other kind).
    """
    if data.startswith(b"%PDF-"):
        return "this is a PDF, extract its text first (e.g. pdftotext or PyMuPDF)"
    if data.startswith(b"PK\x03\x04"):
        return (
            "this is a zip container (docx/epub/pptx), extract its text first "
            "(e.g. pandoc)"
        )
    if b"\x00" in data[:_SNIFF_BYTES]:
        return "binary file (contains a null byte); extract plain text first"
    return None


def _chunk_count(lines: list[str], chunk_lines: int) -> int:
    """Count the non-empty ``chunk_lines``-line chunks a source yields.

    Mirrors :func:`curriculum.app.build._ingest_source` exactly: lines are grouped
    into ``chunk_lines``-sized windows and a window is dropped when its joined text
    is blank (all whitespace), so the estimate matches the real extract-call count
    rather than an optimistic ceil division.
    """
    count = 0
    for start in range(0, len(lines), chunk_lines):
        text = "\n".join(lines[start : start + chunk_lines]).strip()
        if text:
            count += 1
    return count

"""File-backed :class:`ContentRepository` over an OKF v0.1 markdown bundle.

This is the on-disk counterpart to ``InMemoryContentRepository``: the prose
content of concepts and questions lives as markdown files with YAML frontmatter
(see docs/okf-spec.md), and Postgres holds only the structural index. The two
adapters are interchangeable behind the ``ContentRepository`` port.

Layout
------
* A concept ``c`` is stored at ``<root>/<c>.md``. The concept id is the bundle
  path with the ``.md`` suffix removed (spec section 2), so ids may contain ``/``
  and map onto nested directories, which we create on write.
* The frontmatter carries ``type: Concept`` plus the recommended ``title`` and
  ``description`` fields. The body is the concept prose; any ``source_refs`` are
  rendered as a trailing ``# Citations`` section (spec section 8) and parsed back
  on read, so the round-trip is lossless.
* A question ``q`` is stored at ``<root>/questions/<q>.md`` with
  ``type: Question`` and ``# Prompt`` / ``# Rubric`` body sections.

Hashing
-------
``put_*`` return the SHA-256 of the *file bytes* actually written (not a hash of
the dataclass), so the sync layer can compare it against what is on disk; the
identical digest is reported by :meth:`iter_concepts` for the same file.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterable

from ..domain.entities import ConceptContent, QuestionContent, SourceRef
from ..ports.repositories import ContentRepository
from ..okf import frontmatter

__all__ = ["FileContentRepository"]

# Reserved filenames that are never concept documents (spec section 3.1), plus
# the questions subtree which holds question docs, not concepts.
_RESERVED = frozenset({"index.md", "log.md"})
_QUESTIONS_DIR = "questions"

# A citations section appended after the body. ``body`` is greedy so the LAST
# ``# Citations`` fence is treated as the boundary, which is robust to a body
# that happens to mention the heading earlier.
_CITATIONS_RE = re.compile(
    r"\A(?P<body>.*)\n\n# Citations\n\n(?P<cites>.*)\Z", re.DOTALL
)
# A single numbered citation line: ``[1] some/ref.md:42``.
_CITATION_LINE_RE = re.compile(r"^\[\d+\]\s+(?P<target>.*\S)\s*$")

# Prompt/rubric body sections for a question document. ``prompt`` is non-greedy
# so the optional rubric section is captured separately when present.
_QUESTION_RE = re.compile(
    r"\A# Prompt\n\n(?P<prompt>.*?)(?:\n\n# Rubric\n\n(?P<rubric>.*))?\Z",
    re.DOTALL,
)


class FileContentRepository(ContentRepository):
    """Read/write concept and question prose as files under a bundle *root*."""

    def __init__(self, root: Path) -> None:
        # Coerce to Path so a plain string root is accepted too.
        self._root = Path(root)

    # ------------------------------------------------------------------ #
    # Concepts
    # ------------------------------------------------------------------ #
    def get_concept_content(self, concept_id: str) -> ConceptContent | None:
        path = self._concept_path(concept_id)
        if not path.is_file():
            return None
        meta, file_body = frontmatter.parse(path.read_text(encoding="utf-8"))
        body, source_refs = _split_citations(file_body)
        # Per the spec, derive a title from the basename only when the field is
        # genuinely absent; a present-but-empty title is preserved verbatim.
        title = meta.get("title", concept_id.rsplit("/", 1)[-1])
        return ConceptContent(
            concept_id=concept_id,
            title=str(title),
            body=body,
            description=str(meta.get("description", "")),
            source_refs=source_refs,
        )

    def put_concept_content(self, content: ConceptContent) -> str:
        meta = {
            "type": "Concept",
            "title": content.title,
            "description": content.description,
        }
        file_body = _compose_concept_body(content.body, content.source_refs)
        return self._write(self._concept_path(content.concept_id), meta, file_body)

    # ------------------------------------------------------------------ #
    # Questions
    # ------------------------------------------------------------------ #
    def get_question_content(self, question_id: str) -> QuestionContent | None:
        path = self._question_path(question_id)
        if not path.is_file():
            return None
        _meta, file_body = frontmatter.parse(path.read_text(encoding="utf-8"))
        prompt, rubric = _split_question_body(file_body)
        return QuestionContent(question_id=question_id, prompt=prompt, rubric=rubric)

    def put_question_content(self, content: QuestionContent) -> str:
        meta = {"type": "Question"}
        file_body = _compose_question_body(content.prompt, content.rubric)
        return self._write(self._question_path(content.question_id), meta, file_body)

    # ------------------------------------------------------------------ #
    # Enumeration
    # ------------------------------------------------------------------ #
    def iter_concepts(self) -> Iterable[tuple[str, str]]:
        """Yield ``(concept_id, sha256)`` for every concept doc in the bundle.

        Walks the whole tree but skips the reserved ``index.md`` / ``log.md``
        files and the entire ``questions/`` subtree (those are not concepts).
        Results are sorted by concept id so enumeration is deterministic across
        platforms and filesystems."""
        if not self._root.is_dir():
            return
        found: list[tuple[str, str]] = []
        for path in self._root.rglob("*.md"):
            if path.name in _RESERVED:
                continue
            rel = path.relative_to(self._root)
            if rel.parts and rel.parts[0] == _QUESTIONS_DIR:
                continue
            concept_id = rel.with_suffix("").as_posix()
            found.append((concept_id, _sha256(path.read_bytes())))
        found.sort(key=lambda pair: pair[0])
        yield from found

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _concept_path(self, concept_id: str) -> Path:
        # ``concept_id`` uses POSIX ``/`` separators; joining yields the nested
        # on-disk path regardless of the host separator.
        return self._root.joinpath(*f"{concept_id}.md".split("/"))

    def _question_path(self, question_id: str) -> Path:
        return self._root.joinpath(_QUESTIONS_DIR, *f"{question_id}.md".split("/"))

    def _write(self, path: Path, meta: dict, file_body: str) -> str:
        """Write the document and return the SHA-256 of the bytes written."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = frontmatter.dump(meta, file_body).encode("utf-8")
        path.write_bytes(data)
        return _sha256(data)


# --------------------------------------------------------------------------- #
# Body <-> structured-content helpers (module-level: pure, easy to test)
# --------------------------------------------------------------------------- #
def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _compose_concept_body(body: str, source_refs: tuple[SourceRef, ...]) -> str:
    """Append a numbered ``# Citations`` section iff there are source refs."""
    if not source_refs:
        return body
    cites = "\n".join(
        f"[{i}] {_render_ref(ref)}" for i, ref in enumerate(source_refs, start=1)
    )
    return f"{body}\n\n# Citations\n\n{cites}"


def _split_citations(file_body: str) -> tuple[str, tuple[SourceRef, ...]]:
    """Inverse of :func:`_compose_concept_body`."""
    match = _CITATIONS_RE.match(file_body)
    if match is None:
        return file_body, ()
    refs = tuple(
        _parse_ref(m.group("target"))
        for line in match.group("cites").split("\n")
        if (m := _CITATION_LINE_RE.match(line.strip()))
    )
    return match.group("body"), refs


def _render_ref(ref: SourceRef) -> str:
    """Render a source ref as ``file`` or ``file:line``."""
    return ref.file if ref.line is None else f"{ref.file}:{ref.line}"


def _parse_ref(target: str) -> SourceRef:
    """Inverse of :func:`_render_ref`.

    The trailing ``:<n>`` is read as a line number only when ``<n>`` is all
    digits, so a colon-bearing path with no line (e.g. a URL) stays intact."""
    head, sep, tail = target.rpartition(":")
    if sep and head and tail.isdigit():
        return SourceRef(file=head, line=int(tail))
    return SourceRef(file=target, line=None)


def _compose_question_body(prompt: str, rubric: str) -> str:
    """Render the prompt and (optional) rubric as markdown sections."""
    body = f"# Prompt\n\n{prompt}"
    if rubric:
        body += f"\n\n# Rubric\n\n{rubric}"
    return body


def _split_question_body(file_body: str) -> tuple[str, str]:
    """Inverse of :func:`_compose_question_body`.

    A document we did not author (no ``# Prompt`` fence) degrades gracefully:
    the whole body becomes the prompt and the rubric is empty."""
    match = _QUESTION_RE.match(file_body)
    if match is None:
        return file_body, ""
    rubric = match.group("rubric")
    return match.group("prompt"), rubric if rubric is not None else ""

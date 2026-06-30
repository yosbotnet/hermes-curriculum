"""One-way, hash-keyed reconciliation: OKF bundle -> Postgres concept index.

The OKF markdown bundle is the source of truth for concept PROSE; the Postgres
``ConceptIndexRepository`` holds the derived STRUCTURE/METADATA index plus a
content embedding cache (see ``domain/entities.py`` for the OKF/Postgres split).
Those two stores drift apart whenever an author edits the bundle, so this
service brings the index back in line with the bundle.

Why one-way (bundle -> index, never the reverse): the bundle is authoritative,
so reconciliation only ever READS content and WRITES the index. It never calls a
``put_*`` on the content repository. This keeps the dataflow a pure projection
and removes any chance of the staleness marker writing back over the source.

Why hash-keyed: every concept carries a ``content_hash`` (sha256 of its OKF
doc). Comparing the bundle's current hash against the hash recorded on the index
row classifies each concept in O(1) without reading the body -- so an unchanged
concept costs one dict lookup and, crucially, is NOT re-embedded (embeddings are
the only expensive step, and the embedder is the only paid dependency here).

Determinism: the service performs no I/O of its own and reads no clock; the
order of work is fixed by ``content.iter_concepts()`` (a stable bundle walk) and
``index.list_by_course()`` (a stable index scan), so a report is reproducible.
Standard library only.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping

from ..domain.entities import Concept, ConceptContent
from ..domain.errors import SyncError
from ..ports.providers import EmbeddingProvider
from ..ports.repositories import ConceptIndexRepository, ContentRepository

__all__ = ["SyncReport", "GraphSyncService"]


@dataclass(frozen=True, slots=True)
class SyncReport:
    """Outcome of one :meth:`GraphSyncService.reconcile` pass.

    Each list holds the concept ids that fell into that bucket, in the
    deterministic order they were processed (bundle order for the first three,
    index order for orphans). The buckets partition every concept seen exactly
    once, so they never overlap. Count properties are derived rather than stored
    to keep a single source of truth (the lists themselves)."""

    added: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    orphaned: list[str] = field(default_factory=list)

    @property
    def n_added(self) -> int:
        return len(self.added)

    @property
    def n_changed(self) -> int:
        return len(self.changed)

    @property
    def n_unchanged(self) -> int:
        return len(self.unchanged)

    @property
    def n_orphaned(self) -> int:
        return len(self.orphaned)

    @property
    def total(self) -> int:
        """Total concepts touched across all four buckets."""
        return self.n_added + self.n_changed + self.n_unchanged + self.n_orphaned

    @property
    def counts(self) -> Mapping[str, int]:
        """Bucket -> size, handy for a one-line log of a sync run."""
        return {
            "added": self.n_added,
            "changed": self.n_changed,
            "unchanged": self.n_unchanged,
            "orphaned": self.n_orphaned,
        }


class GraphSyncService:
    """Reconcile the Postgres concept index against the OKF content bundle.

    The service is a stateless projection over three ports; it owns no data, so
    a fresh report is produced on every :meth:`reconcile` call and nothing
    persists on the instance between runs apart from its injected collaborators.
    """

    def __init__(
        self,
        content: ContentRepository,
        index: ConceptIndexRepository,
        embedder: EmbeddingProvider,
        *,
        course: str,
        delete_orphans: bool = False,
    ) -> None:
        self._content = content
        self._index = index
        self._embedder = embedder
        self._course = course
        self._delete_orphans = delete_orphans

    def reconcile(self) -> SyncReport:
        """Bring the index in line with the bundle and report what changed.

        Pass 1 walks the bundle and classifies each concept by comparing the
        bundle hash with the hash on the index row (added / changed / unchanged).
        Pass 2 scans the index for this course and flags any row the bundle no
        longer contains as orphaned, deleting it when ``delete_orphans`` is set.
        """
        report = SyncReport()
        seen: set[str] = set()

        for concept_id, content_hash in self._content.iter_concepts():
            seen.add(concept_id)
            existing = self._index.get(concept_id)
            if existing is None:
                # Never indexed before: project content -> index row and embed.
                self._project_and_embed(concept_id, content_hash, None)
                report.added.append(concept_id)
            elif existing.content_hash != content_hash:
                # Body moved on (hash differs): refresh the row and RE-embed,
                # preserving Postgres-authoritative metadata (importance/status).
                self._project_and_embed(concept_id, content_hash, existing)
                report.changed.append(concept_id)
            else:
                # Hash matches: the body is identical, so skip the embedder.
                report.unchanged.append(concept_id)

        self._collect_orphans(seen, report)
        return report

    def _project_and_embed(
        self, concept_id: str, content_hash: str, existing: Concept | None
    ) -> None:
        """Upsert the index row for ``concept_id`` and (re)compute its embedding.

        The content body is the source of the embedding, so this is the only
        path that loads :class:`ConceptContent`; it is reached exactly when a
        concept is new or its hash changed. A missing body for a hash that the
        bundle just advertised is a corrupt bundle, not a normal absence, so we
        fail loudly with :class:`SyncError` rather than embedding ``None``."""
        content = self._content.get_concept_content(concept_id)
        if content is None:
            raise SyncError(
                f"bundle advertised concept {concept_id!r} but its content is missing"
            )
        self._index.upsert(self._project(concept_id, content_hash, content, existing))
        # embed() is batch-oriented; we hand it one body and take the one vector.
        vector = self._embedder.embed([content.body])[0]
        self._index.set_embedding(concept_id, vector)

    def _project(
        self,
        concept_id: str,
        content_hash: str,
        content: ConceptContent,
        existing: Concept | None,
    ) -> Concept:
        """Build the index row from OKF content plus the fresh hash.

        For a brand-new concept we lean on :class:`Concept`'s defaults. For an
        existing one we ``replace`` only the content-derived fields (title,
        description, source_refs), the course, and the hash -- leaving
        importance/status untouched because those are curated in Postgres and
        have no counterpart in the OKF body, so a content edit must not reset
        them."""
        if existing is None:
            return Concept(
                id=concept_id,
                course=self._course,
                title=content.title,
                description=content.description,
                source_refs=content.source_refs,
                content_hash=content_hash,
            )
        return replace(
            existing,
            course=self._course,
            title=content.title,
            description=content.description,
            source_refs=content.source_refs,
            content_hash=content_hash,
        )

    def _collect_orphans(self, seen: set[str], report: SyncReport) -> None:
        """Flag (and optionally delete) index rows the bundle no longer has.

        An orphan is a concept that exists in the index for this course but was
        not produced by the current bundle walk. Deletion is gated behind
        ``delete_orphans`` so a default run is observe-only: it reports drift
        without destroying index state, which is the safe choice when a bundle
        path might merely have been moved or temporarily withheld."""
        for concept in self._index.list_by_course(self._course):
            if concept.id in seen:
                continue
            report.orphaned.append(concept.id)
            if self._delete_orphans:
                self._index.delete(concept.id)

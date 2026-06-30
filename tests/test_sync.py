"""Tests for the one-way OKF -> index reconciliation (``GraphSyncService``).

Stdlib unittest only. The service does no I/O and reads no clock, so every run
is deterministic and these tests need no injected randomness or fixed time.

We assemble the real in-memory adapters (``InMemoryContentRepository`` is the
executable spec of the OKF bundle; ``InMemoryConceptIndexRepository`` of the
Postgres index) and wrap the deterministic ``FakeEmbedder`` in a call-counting
proxy so we can assert exactly when the (expensive) embedder is and is NOT
invoked -- the central efficiency invariant of the hash-keyed design.
"""
from __future__ import annotations

import unittest
from typing import Iterable, Sequence

from curriculum.domain.entities import Concept, ConceptContent, SourceRef
from curriculum.domain.errors import SyncError
from curriculum.providers_fake import FakeEmbedder
from curriculum.ports.providers import EmbeddingProvider
from curriculum.ports.repositories import ContentRepository
from curriculum.storage.memory import (
    InMemoryConceptIndexRepository,
    InMemoryContentRepository,
)
from curriculum.sync.graph_sync import GraphSyncService, SyncReport

COURSE = "cs101"


class CountingEmbedder(EmbeddingProvider):
    """Wrap a real embedder and record every text it is asked to embed.

    ``calls`` is the flat list of all texts passed to ``embed`` across every
    invocation, so a test can assert both how many times and with what body the
    embedder ran. Delegating to a genuine :class:`FakeEmbedder` keeps the
    vectors meaningful (distinct bodies -> distinct vectors) for nearest checks.
    """

    def __init__(self, inner: EmbeddingProvider) -> None:
        self._inner = inner
        self.dim = inner.dim
        self.calls: list[str] = []

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.extend(texts)
        return self._inner.embed(texts)


class RemovableContent(InMemoryContentRepository):
    """In-memory content repo that can also drop a concept from the bundle.

    The production port has no delete (authoring removes a file on disk); for
    orphan tests we expose ``remove`` so a previously synced concept can vanish
    from ``iter_concepts`` exactly as a deleted markdown file would.
    """

    def remove(self, concept_id: str) -> None:
        self._concept_content.pop(concept_id, None)
        self._concept_hash.pop(concept_id, None)


class ReadGuardContent(ContentRepository):
    """Delegate reads to an inner repo; make any WRITE explode.

    Used to prove the sync is strictly one-way: if ``reconcile`` ever called a
    ``put_*`` on the content store, this proxy would raise and fail the test.
    """

    def __init__(self, inner: ContentRepository) -> None:
        self._inner = inner

    def get_concept_content(self, concept_id: str) -> ConceptContent | None:
        return self._inner.get_concept_content(concept_id)

    def put_concept_content(self, content: ConceptContent) -> str:  # noqa: D102
        raise AssertionError("sync must not write content (one-way only)")

    def get_question_content(self, question_id):  # type: ignore[override]
        return self._inner.get_question_content(question_id)

    def put_question_content(self, content):  # type: ignore[override]
        raise AssertionError("sync must not write content (one-way only)")

    def iter_concepts(self) -> Iterable[tuple[str, str]]:
        return self._inner.iter_concepts()


class MissingBodyContent(ContentRepository):
    """A corrupt bundle: ``iter_concepts`` advertises an id whose body is gone."""

    def get_concept_content(self, concept_id: str) -> ConceptContent | None:
        return None

    def put_concept_content(self, content: ConceptContent) -> str:
        raise NotImplementedError

    def get_question_content(self, question_id):  # type: ignore[override]
        return None

    def put_question_content(self, content):  # type: ignore[override]
        raise NotImplementedError

    def iter_concepts(self) -> Iterable[tuple[str, str]]:
        yield "ghost", "deadbeef"


def _content(cid: str, body: str, *, title: str | None = None) -> ConceptContent:
    """Build a ConceptContent with a couple of grounding refs for metadata checks."""
    return ConceptContent(
        concept_id=cid,
        title=title or f"Title {cid}",
        body=body,
        description=f"desc {cid}",
        source_refs=(SourceRef(file=f"{cid}.md", line=1),),
    )


class GraphSyncTests(unittest.TestCase):
    def setUp(self) -> None:
        self.content = RemovableContent()
        self.index = InMemoryConceptIndexRepository()
        self.embedder = CountingEmbedder(FakeEmbedder(dim=16))

    def _service(self, *, delete_orphans: bool = False) -> GraphSyncService:
        return GraphSyncService(
            self.content,
            self.index,
            self.embedder,
            course=COURSE,
            delete_orphans=delete_orphans,
        )

    # -- added ------------------------------------------------------------- #
    def test_new_concept_is_added_and_embedded(self) -> None:
        self.content.put_concept_content(_content("c1", "alpha body"))

        report = self._service().reconcile()

        self.assertEqual(report.added, ["c1"])
        self.assertEqual(report.changed, [])
        self.assertEqual(report.unchanged, [])
        self.assertEqual(report.orphaned, [])
        # The index row was written with the bundle hash + course + metadata.
        stored = self.index.get("c1")
        self.assertIsNotNone(stored)
        assert stored is not None
        self.assertEqual(stored.course, COURSE)
        self.assertEqual(stored.title, "Title c1")
        self.assertEqual(stored.description, "desc c1")
        self.assertEqual(stored.source_refs, (SourceRef(file="c1.md", line=1),))
        self.assertIsNotNone(stored.content_hash)
        # The body was embedded exactly once and is now nearest-searchable.
        self.assertEqual(self.embedder.calls, ["alpha body"])
        query = self.embedder.embed(["alpha body"])[0]
        nearest = self.index.nearest(query, course=COURSE, k=1)
        self.assertEqual(nearest[0][0], "c1")

    # -- unchanged --------------------------------------------------------- #
    def test_unchanged_concept_is_skipped_without_reembedding(self) -> None:
        self.content.put_concept_content(_content("c1", "alpha body"))
        self._service().reconcile()
        calls_after_first = len(self.embedder.calls)

        report = self._service().reconcile()

        self.assertEqual(report.unchanged, ["c1"])
        self.assertEqual(report.added, [])
        self.assertEqual(report.changed, [])
        # KEY INVARIANT: an unchanged concept does not touch the embedder again.
        self.assertEqual(len(self.embedder.calls), calls_after_first)

    # -- changed ----------------------------------------------------------- #
    def test_changed_body_is_reembedded_and_hash_updated(self) -> None:
        self.content.put_concept_content(_content("c1", "alpha body"))
        first = self._service().reconcile()
        old_hash = self.index.get("c1").content_hash
        calls_after_first = len(self.embedder.calls)

        # Author edits the body -> new hash advertised by the bundle.
        self.content.put_concept_content(_content("c1", "BETA body rewritten"))
        report = self._service().reconcile()

        self.assertEqual(first.added, ["c1"])
        self.assertEqual(report.changed, ["c1"])
        self.assertEqual(report.added, [])
        self.assertEqual(report.unchanged, [])
        new_hash = self.index.get("c1").content_hash
        self.assertNotEqual(new_hash, old_hash)
        # Re-embedded: one extra call, and on the NEW body text.
        self.assertEqual(len(self.embedder.calls), calls_after_first + 1)
        self.assertEqual(self.embedder.calls[-1], "BETA body rewritten")

    def test_changed_preserves_postgres_authored_metadata(self) -> None:
        # importance/status live in Postgres, not OKF; a content edit must keep
        # them. Simulate a curator having set importance after the first sync.
        self.content.put_concept_content(_content("c1", "alpha body"))
        self._service().reconcile()
        curated = self.index.get("c1")
        assert curated is not None
        self.index.upsert(
            Concept(
                id=curated.id,
                course=curated.course,
                title=curated.title,
                description=curated.description,
                importance=0.95,
                source_refs=curated.source_refs,
                content_hash=curated.content_hash,
                status="frozen",
            )
        )

        self.content.put_concept_content(
            _content("c1", "rewritten", title="New Title")
        )
        report = self._service().reconcile()

        self.assertEqual(report.changed, ["c1"])
        updated = self.index.get("c1")
        assert updated is not None
        # Content-derived fields refreshed...
        self.assertEqual(updated.title, "New Title")
        # ...but curated metadata survived the content edit.
        self.assertEqual(updated.importance, 0.95)
        self.assertEqual(updated.status, "frozen")

    # -- orphans ----------------------------------------------------------- #
    def test_orphan_deleted_when_flag_set(self) -> None:
        self.content.put_concept_content(_content("c1", "one"))
        self.content.put_concept_content(_content("c2", "two"))
        self._service().reconcile()

        # c2's file is removed from the bundle.
        self.content.remove("c2")
        report = self._service(delete_orphans=True).reconcile()

        self.assertEqual(report.orphaned, ["c2"])
        self.assertEqual(report.unchanged, ["c1"])
        self.assertIsNone(self.index.get("c2"))
        self.assertIsNotNone(self.index.get("c1"))

    def test_orphan_reported_but_kept_when_flag_unset(self) -> None:
        self.content.put_concept_content(_content("c1", "one"))
        self.content.put_concept_content(_content("c2", "two"))
        self._service().reconcile()

        self.content.remove("c2")
        report = self._service(delete_orphans=False).reconcile()

        self.assertEqual(report.orphaned, ["c2"])
        # Default run is observe-only: the row is still there.
        self.assertIsNotNone(self.index.get("c2"))

    def test_orphan_detection_is_scoped_to_the_courses_own_rows(self) -> None:
        # A concept belonging to another course must never be flagged as an
        # orphan of this course's bundle.
        self.index.upsert(Concept(id="other", course="other_course", title="x"))
        self.content.put_concept_content(_content("c1", "one"))

        report = self._service(delete_orphans=True).reconcile()

        self.assertEqual(report.orphaned, [])
        self.assertIsNotNone(self.index.get("other"))

    # -- report shape ------------------------------------------------------ #
    def test_report_counts_match_list_lengths(self) -> None:
        self.content.put_concept_content(_content("a", "aa"))
        self.content.put_concept_content(_content("b", "bb"))
        report = self._service().reconcile()  # both added on first pass
        self.assertEqual(report.n_added, 2)
        self.assertEqual(report.counts, {"added": 2, "changed": 0, "unchanged": 0, "orphaned": 0})
        self.assertEqual(report.total, 2)

    def test_empty_report_defaults(self) -> None:
        report = SyncReport()
        self.assertEqual(report.added, [])
        self.assertEqual(report.total, 0)

    def test_full_run_classifies_each_bucket_once(self) -> None:
        # Set up: c_keep (unchanged), c_edit (changed), c_drop (orphan); then a
        # brand-new c_new is added. Verifies the buckets partition cleanly.
        self.content.put_concept_content(_content("c_keep", "keep"))
        self.content.put_concept_content(_content("c_edit", "edit-v1"))
        self.content.put_concept_content(_content("c_drop", "drop"))
        self._service().reconcile()

        self.content.put_concept_content(_content("c_edit", "edit-v2"))
        self.content.remove("c_drop")
        self.content.put_concept_content(_content("c_new", "new"))
        report = self._service(delete_orphans=True).reconcile()

        self.assertEqual(report.added, ["c_new"])
        self.assertEqual(report.changed, ["c_edit"])
        self.assertEqual(report.unchanged, ["c_keep"])
        self.assertEqual(report.orphaned, ["c_drop"])

    # -- invariants -------------------------------------------------------- #
    def test_sync_never_writes_to_content_store(self) -> None:
        inner = RemovableContent()
        inner.put_concept_content(_content("c1", "alpha"))
        guarded = ReadGuardContent(inner)
        service = GraphSyncService(
            guarded, self.index, self.embedder, course=COURSE
        )
        # Would raise AssertionError inside ReadGuardContent if it wrote back.
        report = service.reconcile()
        self.assertEqual(report.added, ["c1"])

    def test_missing_body_for_advertised_hash_raises_sync_error(self) -> None:
        service = GraphSyncService(
            MissingBodyContent(), self.index, self.embedder, course=COURSE
        )
        with self.assertRaises(SyncError):
            service.reconcile()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

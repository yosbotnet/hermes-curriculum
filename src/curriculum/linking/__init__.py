"""Post-ingest graph repair: connect concepts the inference pass left isolated.

The single public export is :class:`EmbeddingLinker`, the embedding-guided
anti-isolation step. It is kept in its own subpackage (rather than as a sixth
ingestion pass) because it runs AFTER persistence over the final graph, reusing
the stored pgvector cache through the repository ports -- it is not part of the
in-memory ingestion working set.
"""
from __future__ import annotations

from .embedding_linker import EmbeddingLinker

__all__ = ["EmbeddingLinker"]

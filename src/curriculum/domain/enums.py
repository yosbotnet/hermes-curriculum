"""Closed vocabularies used across the domain.

Kept tiny and stable: these are part of the public contract that adapters and
strategies depend on, so changes here ripple widely. Add values conservatively.
"""
from __future__ import annotations

from enum import Enum, IntEnum


class EdgeType(str, Enum):
    """The three relationship kinds in the knowledge graph.

    - PREREQUISITE: src must be mastered before dst is learnable (gating).
    - ENCOMPASSES: practising src implicitly exercises dst (FIRe credit travels
      down this edge; weight = fraction of dst exercised, 0..1).
    - RELATED: a non-gating connection between concepts (the unit of
      connection-skip tracking).
    """

    PREREQUISITE = "prerequisite"
    ENCOMPASSES = "encompasses"
    RELATED = "related"


class Mastery(str, Enum):
    """Criterion-based progression for a concept (see SR-Cybersecurity.md legend)."""

    NEW = "new"
    LEARNING = "learning"
    SOLID = "solid"
    EXAM_READY = "exam_ready"


class FsrsRating(IntEnum):
    """The four-point rating an FSRS-style scheduler consumes.

    The application maps a 0..6 question grade onto one of these (see the
    grade->rating mapping in the course profile).
    """

    AGAIN = 1
    HARD = 2
    GOOD = 3
    EASY = 4


class NextMode(str, Enum):
    """What the engine wants the tutor to do next with a concept."""

    TEACH = "teach"      # introduce a new (learnable) concept
    REVIEW = "review"    # spaced-repetition review of a known concept
    TEST = "test"        # forced retrieval (e.g. an escalated skipped connection)

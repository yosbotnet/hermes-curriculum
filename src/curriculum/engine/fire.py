"""Fractional Implicit Repetition (FIRe) credit propagation.

Math Academy's FIRe insight: practising a concept implicitly exercises the
sub-skills it ENCOMPASSES, so a single graded answer should ripple credit (or
blame) onto neighbouring concepts instead of being recorded in isolation. This
module turns one ``GradeRecorded`` event into the set of *implicit* rating
updates the grade() use-case must additionally apply.

Two strategies behind the same port:

* ``NoPropagation`` -- the Null Object. It returns nothing, so the engine's
  grading path is byte-for-byte identical whether FIRe is enabled or not. This
  keeps the FIRe feature flag free of branching in the caller (Open/Closed).
* ``FirePropagation`` -- the real algorithm over ENCOMPASSES edges.

Direction matters and is asymmetric on purpose:

* A clear PASS spreads *positive* credit DOWN the ENCOMPASSES edges (from the
  broad concept to the narrower sub-skills it exercises). Credit decays with
  the edge weight (the fraction of the child actually exercised) and with each
  extra hop, because evidence gets thinner the further it travels.
* A FAIL spreads *blame* UP to the direct parents that claim to encompass the
  failed concept: if you cannot do the sub-skill, the parent's mastery is
  suspect too. Penalties travel up, credit travels down.

Everything here is pure, deterministic and stdlib-only: no clock, no RNG, no
I/O beyond the injected ``EdgeRepository``.
"""
from __future__ import annotations

from typing import Sequence

from ..domain.enums import EdgeType, FsrsRating
from ..domain.entities import Edge
from ..domain.events import GradeRecorded
from ..ports.repositories import EdgeRepository
from ..ports.strategies import CreditPropagationStrategy


class NoPropagation(CreditPropagationStrategy):
    """Null Object: FIRe disabled.

    Returns an empty sequence so the caller never needs an ``if fire_enabled``
    branch -- the engine path is identical whether propagation is on or off.
    """

    def propagate(self, event: GradeRecorded) -> Sequence[tuple[str, FsrsRating]]:
        """No implicit updates. Always the empty tuple."""
        return ()


def _down_rating_for_weight(weight: float) -> FsrsRating | None:
    """Map an ENCOMPASSES edge weight to the base implicit PASS rating.

    The weight is the fraction of the child concept exercised by the parent, so
    a strong link earns the child a near-equivalent of a real GOOD review while
    a thin link earns only a HARD nudge. Below the floor the evidence is too
    weak to count as a repetition at all (the child gets nothing).
    """
    if weight >= 0.66:
        return FsrsRating.GOOD
    if weight >= 0.33:
        return FsrsRating.HARD
    return None


def _discounted(rating: FsrsRating, hops: int) -> FsrsRating | None:
    """Lower ``rating`` by one tier for every hop beyond the first.

    ``hops`` counts edges traversed from the origin (1 == a direct child). The
    first hop is undiscounted; each additional hop drops one tier because the
    implicit evidence weakens the further it propagates. Once it would fall
    below HARD we return None: anything weaker is not worth recording as a
    repetition, and -- since deeper hops only discount further -- this also
    tells the caller it can safely stop recursing down this branch.
    """
    value = int(rating) - (hops - 1)
    if value < int(FsrsRating.HARD):
        return None
    return FsrsRating(value)


class FirePropagation(CreditPropagationStrategy):
    """Fractional Implicit Repetition over the ENCOMPASSES sub-graph."""

    def __init__(self, edges: EdgeRepository, *, max_depth: int = 2) -> None:
        """``edges`` is the knowledge-graph edge store; ``max_depth`` bounds how
        many ENCOMPASSES hops PASS credit may travel (keeps propagation cheap
        and guarantees termination even on cyclic graphs)."""
        self._edges = edges
        self.max_depth = max_depth

    def propagate(self, event: GradeRecorded) -> Sequence[tuple[str, FsrsRating]]:
        """Return the implicit (concept_id, rating) updates implied by ``event``.

        The event's own concept is never included (it is updated explicitly by
        the scheduler). Duplicates -- a concept reached by several paths -- are
        collapsed keeping the strongest rating. The result is sorted by
        concept_id so the output is fully deterministic regardless of edge
        iteration order.
        """
        emissions: list[tuple[str, FsrsRating]] = []

        # Speed guardrail: only a clear pass (GOOD or better) spreads credit
        # down. A borderline HARD answer is too ambiguous to vouch for the
        # sub-skills, so it propagates nothing.
        if event.rating >= FsrsRating.GOOD:
            self._spread_down(
                event.concept_id, 0, emissions, frozenset({event.concept_id})
            )
        elif event.rating == FsrsRating.AGAIN:
            # Blame travels straight up to the direct parents. AGAIN is already
            # the lowest tier, so there is no tier to discount across hops; we
            # penalise the immediate encompassing concepts only.
            for edge in self._encompasses_in(event.concept_id):
                emissions.append((edge.src, FsrsRating.AGAIN))

        return self._dedup(emissions, exclude=event.concept_id)

    # -- traversal ---------------------------------------------------------- #
    def _spread_down(
        self,
        src: str,
        hops: int,
        emissions: list[tuple[str, FsrsRating]],
        seen_on_path: frozenset[str],
    ) -> None:
        """Depth-first push of PASS credit down ENCOMPASSES out-edges.

        ``hops`` is the number of edges already traversed to reach ``src``;
        each child therefore sits at ``hops + 1``. ``seen_on_path`` carries the
        concepts on the current path so a cycle cannot be re-entered. Both the
        explicit depth cap and the HARD discount floor bound the recursion.
        """
        if hops >= self.max_depth:
            return
        for edge in self._encompasses_out(src):
            dst = edge.dst
            if dst in seen_on_path:
                continue  # cycle guard: never revisit a node on this path
            base = _down_rating_for_weight(edge.weight)
            if base is None:
                continue  # weak link: no credit, and nothing to carry deeper
            rating = _discounted(base, hops + 1)
            if rating is None:
                continue  # decayed below HARD: deeper hops only weaker -> prune
            emissions.append((dst, rating))
            self._spread_down(dst, hops + 1, emissions, seen_on_path | {dst})

    # -- edge access (defensive ENCOMPASSES filter) ------------------------- #
    def _encompasses_out(self, src: str) -> list[Edge]:
        """ENCOMPASSES edges leaving ``src`` (broad -> narrow)."""
        return [
            e
            for e in self._edges.out_edges(src, EdgeType.ENCOMPASSES)
            if e.type is EdgeType.ENCOMPASSES
        ]

    def _encompasses_in(self, dst: str) -> list[Edge]:
        """ENCOMPASSES edges entering ``dst`` (the parents that encompass it)."""
        return [
            e
            for e in self._edges.in_edges(dst, EdgeType.ENCOMPASSES)
            if e.type is EdgeType.ENCOMPASSES
        ]

    # -- post-processing ---------------------------------------------------- #
    @staticmethod
    def _dedup(
        emissions: list[tuple[str, FsrsRating]], *, exclude: str
    ) -> list[tuple[str, FsrsRating]]:
        """Drop ``exclude`` and collapse duplicate concepts to their strongest
        rating, returning a concept_id-sorted list for deterministic output."""
        best: dict[str, FsrsRating] = {}
        for concept_id, rating in emissions:
            if concept_id == exclude:
                continue
            current = best.get(concept_id)
            if current is None or int(rating) > int(current):
                best[concept_id] = rating
        return [(concept_id, best[concept_id]) for concept_id in sorted(best)]

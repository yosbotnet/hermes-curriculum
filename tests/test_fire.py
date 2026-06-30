"""Tests for FIRe credit propagation (engine.fire).

These exercise the FIRe contract end to end with a tiny in-memory edge graph:
PASS spreads weight-scaled credit down, FAIL spreads AGAIN up to parents, the
depth bound and tier discount hold, the origin concept is excluded, duplicates
collapse to the strongest rating, and the Null Object propagates nothing.
"""
from __future__ import annotations

import unittest
from datetime import datetime, timezone

from curriculum.domain.entities import Edge
from curriculum.domain.enums import EdgeType, FsrsRating
from curriculum.domain.events import GradeRecorded
from curriculum.ports.repositories import EdgeRepository
from curriculum.engine.fire import FirePropagation, NoPropagation


# --------------------------------------------------------------------------- #
# A minimal in-memory EdgeRepository for the tests.
# --------------------------------------------------------------------------- #
class StubEdgeRepository(EdgeRepository):
    """Just enough of the edge port to drive propagation: an edge list with
    src/dst/type lookups. Mutating methods are no-ops -- FIRe only reads."""

    def __init__(self, edges: list[Edge]) -> None:
        self._edges = list(edges)

    def upsert(self, edge: Edge) -> None:
        self._edges.append(edge)

    def get(self, src: str, dst: str, type: EdgeType) -> Edge | None:
        for e in self._edges:
            if e.src == src and e.dst == dst and e.type == type:
                return e
        return None

    def out_edges(self, src: str, type: EdgeType | None = None):
        return [e for e in self._edges if e.src == src and (type is None or e.type == type)]

    def in_edges(self, dst: str, type: EdgeType | None = None):
        return [e for e in self._edges if e.dst == dst and (type is None or e.type == type)]

    def list_by_course(self, course: str):
        return list(self._edges)

    def record_exposure(self, src, dst, type, *, skipped, at) -> None:  # noqa: D401
        return None


# --------------------------------------------------------------------------- #
# Builders.
# --------------------------------------------------------------------------- #
AT = datetime(2026, 6, 30, 12, 0, tzinfo=timezone.utc)


def enc(src: str, dst: str, weight: float) -> Edge:
    """An ENCOMPASSES edge (broad src -> narrow dst) of the given weight."""
    return Edge(src=src, dst=dst, type=EdgeType.ENCOMPASSES, weight=weight)


def grade(concept_id: str, rating: FsrsRating) -> GradeRecorded:
    """A GradeRecorded carrying only the fields propagation cares about."""
    return GradeRecorded(concept_id=concept_id, grade=int(rating), rating=rating, at=AT)


class NoPropagationTests(unittest.TestCase):
    def test_returns_empty_tuple_on_pass(self):
        result = NoPropagation().propagate(grade("a", FsrsRating.GOOD))
        self.assertEqual(result, ())

    def test_returns_empty_tuple_on_fail(self):
        result = NoPropagation().propagate(grade("a", FsrsRating.AGAIN))
        self.assertEqual(result, ())

    def test_is_credit_propagation_strategy(self):
        from curriculum.ports.strategies import CreditPropagationStrategy

        self.assertIsInstance(NoPropagation(), CreditPropagationStrategy)


class FirePassTests(unittest.TestCase):
    def test_pass_sends_good_and_hard_down_by_weight_weak_gets_nothing(self):
        repo = StubEdgeRepository(
            [
                enc("A", "X", 0.70),  # strong -> GOOD
                enc("A", "Y", 0.40),  # medium -> HARD
                enc("A", "Z", 0.20),  # weak   -> nothing
            ]
        )
        fire = FirePropagation(repo)
        result = dict(fire.propagate(grade("A", FsrsRating.GOOD)))
        self.assertEqual(result, {"X": FsrsRating.GOOD, "Y": FsrsRating.HARD})
        self.assertNotIn("Z", result)

    def test_weight_boundaries_are_inclusive(self):
        repo = StubEdgeRepository(
            [
                enc("A", "G_hi", 0.66),   # boundary GOOD
                enc("A", "H_hi", 0.65),   # just below -> HARD
                enc("A", "H_lo", 0.33),   # boundary HARD
                enc("A", "N", 0.32),      # just below -> nothing
            ]
        )
        result = dict(FirePropagation(repo).propagate(grade("A", FsrsRating.GOOD)))
        self.assertEqual(result["G_hi"], FsrsRating.GOOD)
        self.assertEqual(result["H_hi"], FsrsRating.HARD)
        self.assertEqual(result["H_lo"], FsrsRating.HARD)
        self.assertNotIn("N", result)

    def test_easy_pass_propagates_like_good(self):
        # The down rating is set by edge weight, not by how far above pass the
        # learner scored: an EASY answer still yields at most GOOD downstream.
        repo = StubEdgeRepository([enc("A", "X", 0.9)])
        result = dict(FirePropagation(repo).propagate(grade("A", FsrsRating.EASY)))
        self.assertEqual(result, {"X": FsrsRating.GOOD})

    def test_depth_bound_and_tier_discount(self):
        # A -> B -> C -> D, all strong. With max_depth=2 only B and C are
        # reached; C is discounted one tier (GOOD at hop 1, HARD at hop 2);
        # D is beyond the depth bound.
        repo = StubEdgeRepository(
            [enc("A", "B", 1.0), enc("B", "C", 1.0), enc("C", "D", 1.0)]
        )
        result = dict(FirePropagation(repo, max_depth=2).propagate(grade("A", FsrsRating.GOOD)))
        self.assertEqual(result, {"B": FsrsRating.GOOD, "C": FsrsRating.HARD})
        self.assertNotIn("D", result)

    def test_max_depth_one_stops_at_direct_children(self):
        repo = StubEdgeRepository([enc("A", "B", 1.0), enc("B", "C", 1.0)])
        result = dict(FirePropagation(repo, max_depth=1).propagate(grade("A", FsrsRating.GOOD)))
        self.assertEqual(result, {"B": FsrsRating.GOOD})

    def test_discount_below_hard_is_dropped(self):
        # A HARD-base edge at hop 2 would discount to AGAIN, which is below the
        # repetition floor and must not be emitted as positive credit.
        repo = StubEdgeRepository([enc("A", "B", 1.0), enc("B", "C", 0.40)])
        result = dict(FirePropagation(repo, max_depth=2).propagate(grade("A", FsrsRating.GOOD)))
        self.assertEqual(result, {"B": FsrsRating.GOOD})
        self.assertNotIn("C", result)

    def test_own_concept_excluded_even_with_cycle(self):
        # A <-> B cycle: B gets credit, A (the origin) is never emitted and the
        # cycle does not loop forever.
        repo = StubEdgeRepository([enc("A", "B", 1.0), enc("B", "A", 1.0)])
        result = dict(FirePropagation(repo, max_depth=2).propagate(grade("A", FsrsRating.GOOD)))
        self.assertEqual(result, {"B": FsrsRating.GOOD})
        self.assertNotIn("A", result)

    def test_dedup_keeps_strongest_rating(self):
        # X reachable directly (GOOD) and via M at hop 2 (HARD); the strongest
        # rating wins.
        repo = StubEdgeRepository(
            [enc("A", "X", 0.70), enc("A", "M", 1.0), enc("M", "X", 1.0)]
        )
        result = dict(FirePropagation(repo, max_depth=2).propagate(grade("A", FsrsRating.GOOD)))
        self.assertEqual(result["X"], FsrsRating.GOOD)
        self.assertEqual(result["M"], FsrsRating.GOOD)

    def test_hard_rating_propagates_nothing(self):
        # Speed guardrail: a borderline HARD answer is not a clear pass.
        repo = StubEdgeRepository([enc("A", "X", 1.0)])
        self.assertEqual(list(FirePropagation(repo).propagate(grade("A", FsrsRating.HARD))), [])

    def test_only_encompasses_edges_propagate(self):
        # PREREQUISITE / RELATED out-edges must be ignored even at full weight.
        repo = StubEdgeRepository(
            [
                Edge(src="A", dst="P", type=EdgeType.PREREQUISITE, weight=1.0),
                Edge(src="A", dst="R", type=EdgeType.RELATED, weight=1.0),
                enc("A", "X", 1.0),
            ]
        )
        result = dict(FirePropagation(repo).propagate(grade("A", FsrsRating.GOOD)))
        self.assertEqual(result, {"X": FsrsRating.GOOD})

    def test_result_is_sorted_and_a_list_of_tuples(self):
        repo = StubEdgeRepository([enc("A", "Z", 1.0), enc("A", "B", 1.0), enc("A", "M", 1.0)])
        result = FirePropagation(repo).propagate(grade("A", FsrsRating.GOOD))
        self.assertIsInstance(result, list)
        self.assertEqual([cid for cid, _ in result], ["B", "M", "Z"])
        for item in result:
            self.assertIsInstance(item, tuple)
            self.assertIsInstance(item[1], FsrsRating)


class FireFailTests(unittest.TestCase):
    def test_fail_sends_again_up_to_parents(self):
        # P1 and P2 encompass C; failing C casts blame up to both parents.
        repo = StubEdgeRepository(
            [enc("P1", "C", 0.9), enc("P2", "C", 0.5), enc("C", "child", 1.0)]
        )
        result = dict(FirePropagation(repo).propagate(grade("C", FsrsRating.AGAIN)))
        self.assertEqual(result, {"P1": FsrsRating.AGAIN, "P2": FsrsRating.AGAIN})

    def test_fail_does_not_propagate_down_to_children(self):
        repo = StubEdgeRepository([enc("C", "child", 1.0)])
        result = dict(FirePropagation(repo).propagate(grade("C", FsrsRating.AGAIN)))
        self.assertNotIn("child", result)
        self.assertEqual(result, {})

    def test_fail_blame_is_single_hop(self):
        # Grandparent G encompasses P encompasses C. Failing C blames only the
        # direct parent P, not the grandparent (blame is one hop up).
        repo = StubEdgeRepository([enc("G", "P", 1.0), enc("P", "C", 1.0)])
        result = dict(FirePropagation(repo).propagate(grade("C", FsrsRating.AGAIN)))
        self.assertEqual(result, {"P": FsrsRating.AGAIN})

    def test_fail_ignores_non_encompasses_parents(self):
        repo = StubEdgeRepository(
            [
                Edge(src="P", dst="C", type=EdgeType.PREREQUISITE, weight=1.0),
                enc("Q", "C", 1.0),
            ]
        )
        result = dict(FirePropagation(repo).propagate(grade("C", FsrsRating.AGAIN)))
        self.assertEqual(result, {"Q": FsrsRating.AGAIN})


if __name__ == "__main__":
    unittest.main()

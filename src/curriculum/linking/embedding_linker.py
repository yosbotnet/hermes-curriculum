"""Embedding-guided edge linker: the automatic anti-isolation step.

Edge inference during ingestion (:class:`curriculum.ingestion.passes.InferEdgesPass`)
prompts the model to find edges across the whole concept set in bounded batches.
On large decks that reliably leaves a tail of *isolated* concepts -- nodes with
no in- or out-edges -- either because the model never paired them with anything
or because their batch's JSON got truncated. Isolated concepts are dead weight:
the selection engine can never reach them through the graph, so they are taught
in a vacuum and their connections are never examinable.

The scalable fix (refactored out of ``scripts/repair_emb.py``) is NOT to ask the
model to guess across all N ids again. It is to use the embeddings already cached
in the concept index: for each isolated concept we retrieve its handful of
nearest neighbours by vector similarity (deterministic, cheap, no model), and the
LLM's only job is to CLASSIFY which of those few candidates the concept truly
relates to and with which edge type. Small, grounded prompts -> reliable output.
This is meant to be the standard post-ingest edge step, not a manual patch.

Why a port-based class (not a script): the linker depends only on the repository
and provider ports, so the same code runs against the in-memory adapters under
test and the Postgres adapters in production, and the (paid) LLM is injected --
making the whole step deterministic and unit-testable with a scripted fake.

Standard library only (``json`` for the lenient parse), per the core-module
constraint.
"""
from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from ..domain.entities import Concept, Edge, SourceRef
from ..domain.enums import EdgeType
from ..ports.providers import LlmProvider
from ..ports.repositories import ConceptIndexRepository, EdgeRepository

__all__ = ["EmbeddingLinker"]

# Importance stamped on an inferred repair edge. Set just above the neutral 0.5
# default so a later audit can tell a model-confirmed link from an un-curated
# one, without claiming the exam-relevance of a hand-authored edge.
_INFERRED_IMPORTANCE: float = 0.6

# Output cap for one group's completion. A group can legitimately emit several
# edges per concept, so this is generous; bounding it still guards against a
# runaway model response without truncating a normal answer.
_MAX_TOKENS: int = 4096

# Prompt-trimming budgets: enough signal for the model to judge relatedness,
# short enough to keep each group's prompt small (the whole point of the
# nearest-neighbour pre-filter is to avoid huge prompts).
_CONCEPT_DESC_CHARS: int = 120
_CANDIDATE_DESC_CHARS: int = 60

_LINK_SYSTEM: str = (
    "You classify knowledge-graph edges between an isolated concept and a small "
    "set of its candidate neighbours. For every edge you return, one endpoint "
    "MUST be the isolated concept and the other MUST be one of ITS listed "
    "candidate ids, copied verbatim. Reply with JSON only."
)


def _loads_lenient(text: str) -> Any:
    """Parse JSON that may be wrapped in prose or markdown fences; ``None`` on
    failure.

    Real models often bracket their JSON with explanation or ```` ```json ````
    fences while the test fakes emit bare JSON. We try a strict parse first, then
    fall back to slicing the outermost ``{...}`` / ``[...]`` span. A total failure
    yields ``None`` so the caller treats the completion as "no edges" instead of
    raising and aborting the repair of every other concept in the run.
    """
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = stripped.find(open_ch)
        end = stripped.rfind(close_ch)
        if start != -1 and end > start:
            try:
                return json.loads(stripped[start : end + 1])
            except json.JSONDecodeError:
                continue
    return None


def _edge_dicts(text: str) -> list[dict[str, Any]]:
    """Extract the list of edge objects from a completion.

    Accepts either a bare array or an object wrapping the array under ``edges``
    (some model formattings differ); anything else yields an empty list. Mirrors
    the tolerant JSON contract the ingestion passes use, so the repair step is as
    forgiving of formatting as the rest of the pipeline.
    """
    obj = _loads_lenient(text)
    if isinstance(obj, list):
        return [item for item in obj if isinstance(item, dict)]
    if isinstance(obj, dict):
        value = obj.get("edges")
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


class EmbeddingLinker:
    """Connect isolated concepts to their nearest neighbours via the LLM.

    The linker owns no data: it reads the concept index and the edge graph
    through their ports, asks the injected :class:`LlmProvider` to classify a few
    candidate links per isolated concept, and upserts the edges it accepts. With
    a deterministic embedder behind ``nearest_to`` and a scripted fake LLM, a run
    is fully reproducible.

    Parameters
    ----------
    concepts, edges:
        The repository ports for the concept index and the edge graph.
    llm:
        The completion provider used purely to CLASSIFY candidate edges (never to
        discover them -- that is the embeddings' job).
    k:
        Nearest-neighbour fan-out per isolated concept (candidate pool size).
    batch:
        Number of isolated concepts described per LLM call. Grouping amortises the
        request overhead while keeping each prompt small enough to stay reliable.
    """

    def __init__(
        self,
        concepts: ConceptIndexRepository,
        edges: EdgeRepository,
        llm: LlmProvider,
        *,
        k: int = 10,
        batch: int = 6,
    ) -> None:
        self._concepts = concepts
        self._edges = edges
        self._llm = llm
        # A non-positive fan-out/batch would make the step a silent no-op or
        # divide the work into empty groups; floor both at 1 so the contract
        # ("process in groups of `batch`, k candidates each") always holds.
        self._k = max(1, k)
        self._batch = max(1, batch)

    def link_isolated(self, course: str) -> Mapping[str, Any]:
        """Find isolated concepts in ``course`` and link them to neighbours.

        Returns a small report dict (counts only, JSON-friendly):
        ``isolated_before`` / ``isolated_after`` bracket the run, ``linked`` is
        how many concepts left isolation, ``new_edges`` is how many edges were
        upserted, and ``llm_calls`` is one per processed group (zero when there is
        nothing to do -- the no-isolated case costs no inference).
        """
        concepts = list(self._concepts.list_by_course(course))
        by_id = {c.id: c for c in concepts}
        known_ids = set(by_id)

        isolated = [c for c in concepts if self._is_isolated(c.id)]
        isolated_by_id = {c.id: c for c in isolated}

        # Pre-compute each isolated concept's candidate neighbours ONCE (the
        # embeddings never change during the run), so the prompt builder and the
        # acceptance gate share the same deterministic candidate sets.
        candidates: dict[str, Sequence[tuple[str, float]]] = {
            c.id: self._concepts.nearest_to(c.id, course=course, k=self._k)
            for c in isolated
        }

        # Dedupe accepted edges by their synthetic id across the whole run so two
        # isolated endpoints proposing the same link upsert it (and count it) once.
        accepted: dict[str, Edge] = {}
        llm_calls = 0
        for start in range(0, len(isolated), self._batch):
            group = isolated[start : start + self._batch]
            raw = self._llm.complete(
                self._group_prompt(group, by_id, candidates),
                system=_LINK_SYSTEM,
                temperature=0.0,
                max_tokens=_MAX_TOKENS,
            )
            llm_calls += 1
            group_ids = {c.id for c in group}
            for edge in self._accept(raw, group_ids, candidates, known_ids, isolated_by_id):
                accepted[edge.id] = edge

        for edge in accepted.values():
            self._edges.upsert(edge)

        # Recompute isolation over exactly the once-isolated set: only those
        # concepts could have changed status (every new edge touches one of them).
        isolated_after = sum(1 for c in isolated if self._is_isolated(c.id))
        return {
            "isolated_before": len(isolated),
            "linked": len(isolated) - isolated_after,
            "isolated_after": isolated_after,
            "new_edges": len(accepted),
            "llm_calls": llm_calls,
        }

    def _is_isolated(self, concept_id: str) -> bool:
        """True iff ``concept_id`` has neither an out-edge nor an in-edge.

        Checked through the port (not a cached snapshot) so the post-run recount
        observes the edges this run upserted.
        """
        return not self._edges.out_edges(concept_id) and not self._edges.in_edges(
            concept_id
        )

    def _group_prompt(
        self,
        group: Sequence[Concept],
        by_id: Mapping[str, Concept],
        candidates: Mapping[str, Sequence[tuple[str, float]]],
    ) -> str:
        """Render one classification prompt for a group of isolated concepts.

        Each concept is listed with its candidate neighbours (id + title + a short
        description) so the model judges relatedness from real signal while being
        constrained to pick among the pre-selected ids -- it never has to recall
        an id from the whole graph, which is what kept large-deck output reliable.
        """
        blocks: list[str] = []
        for concept in group:
            lines: list[str] = []
            for neighbour_id, _similarity in candidates.get(concept.id, ()):
                neighbour = by_id.get(neighbour_id)
                title = neighbour.title if neighbour is not None else neighbour_id
                desc = (neighbour.description if neighbour is not None else "")[
                    :_CANDIDATE_DESC_CHARS
                ]
                lines.append(f"    - {neighbour_id}: {title} -- {desc}")
            listing = "\n".join(lines) or "    (no candidate neighbours)"
            blocks.append(
                f"CONCEPT {concept.id}: {concept.title} -- "
                f"{concept.description[:_CONCEPT_DESC_CHARS]}\n"
                f"  candidate neighbours:\n{listing}"
            )
        return (
            "For EACH concept below, decide which of ITS candidate neighbours it "
            "truly relates to and the edge type (prerequisite | encompasses | "
            "related). One endpoint is the concept itself, the other one of ITS "
            "candidate ids -- copy ids verbatim. Connect each concept to at least "
            "its single most-related neighbour.\n\n"
            + "\n\n".join(blocks)
            + '\n\nReturn JSON: {"edges": [{"src": "id", "dst": "id", '
            '"type": "prerequisite|encompasses|related", "rationale": "..."}]}.'
        )

    def _accept(
        self,
        raw: str,
        group_ids: set[str],
        candidates: Mapping[str, Sequence[tuple[str, float]]],
        known_ids: set[str],
        isolated_by_id: Mapping[str, Concept],
    ) -> list[Edge]:
        """Turn one completion into the domain :class:`Edge` objects we keep.

        An inferred edge is kept only when it is anchored to the work at hand:
        exactly one endpoint must be an isolated concept from THIS group, the
        other must be one of that concept's candidate ids (or at least a known
        concept), and it must not be a self-loop. The kept edge is grounded to the
        isolated endpoint's own source file -- so even a repair edge can be traced
        to a real input, preserving the anti-fabrication guarantee. Malformed
        entries and unknown edge types are skipped, never guessed.
        """
        out: list[Edge] = []
        for item in _edge_dicts(raw):
            src = str(item.get("src") or "").strip()
            dst = str(item.get("dst") or "").strip()
            if not src or not dst or src == dst:
                continue
            # Anchor the edge on the isolated concept being processed (prefer src
            # when both endpoints happen to be isolated in this group).
            if src in group_ids:
                iso_end, other = src, dst
            elif dst in group_ids:
                iso_end, other = dst, src
            else:
                continue  # neither endpoint is one of this group's isolated concepts
            candidate_ids = {nid for nid, _ in candidates.get(iso_end, ())}
            if other not in candidate_ids and other not in known_ids:
                continue  # the other endpoint must be a real, related concept
            try:
                edge_type = EdgeType(str(item.get("type") or "").strip().lower())
            except ValueError:
                continue  # unknown relationship kind -> drop rather than guess
            rationale = (str(item.get("rationale") or "").strip()) or None
            out.append(
                Edge(
                    src=src,
                    dst=dst,
                    type=edge_type,
                    importance=_INFERRED_IMPORTANCE,
                    rationale=rationale,
                    source_ref=self._grounding(isolated_by_id[iso_end]),
                )
            )
        return out

    @staticmethod
    def _grounding(concept: Concept) -> SourceRef | None:
        """Ground a repair edge to the isolated endpoint's own first source ref.

        The edge exists because of that concept, so its citation is the honest
        provenance. ``None`` only when the concept itself is somehow ungrounded
        (it should not be, post-VerifyPass), in which case we still record the
        link rather than drop it -- isolation is the worse failure to leave behind.
        """
        return concept.source_refs[0] if concept.source_refs else None

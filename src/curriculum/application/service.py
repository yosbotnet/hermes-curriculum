"""The application service: the use-case orchestration behind the MCP tools.

This is the composition of the engine. It depends ONLY on ports (repositories,
strategies) plus small pure policies -- never on a concrete adapter -- so it is
fully testable with in-memory repositories and the real strategies.

Responsibilities (each kept to a clear method):
- next_action: assemble candidates (due reviews, learnable fringe, escalated
  skipped-connection tests), hand them to the SelectionPolicy, attach a question.
- explain / quiz: fetch grounded content.
- grade: map grade->rating, update the scheduler, run FIRe propagation, update
  connection-skip counts, advance mastery, and log the calibration event.
- state: a burndown snapshot.
"""
from __future__ import annotations

from dataclasses import replace
import re
from datetime import datetime
from typing import Any, Callable, Mapping, Sequence

from ..domain.entities import (
    CandidateContext,
    Concept,
    ConceptContent,
    CourseProfile,
    Edge,
    EngineConfig,
    LearnerState,
    NextAction,
    NextResult,
    Question,
    QuestionContent,
    ReviewEvent,
)
from ..domain.enums import EdgeType, Mastery, NextMode
from ..domain.errors import (
    ConceptNotFound,
    ConfigError,
    ContentNotFound,
    NoCandidatesAvailable,
    QuestionNotFound,
)
from ..domain.events import GradeRecorded
from ..domain.telemetry import EngagementEvent
from ..engine import snapshot
from ..ports.repositories import (
    ConceptIndexRepository,
    ContentRepository,
    CourseProfileRepository,
    EdgeRepository,
    LearnerStateRepository,
    QuestionRepository,
    ReviewLogRepository,
    TelemetryRepository,
)
from ..ports.service import CurriculumService
from ..ports.strategies import CreditPropagationStrategy, SchedulingStrategy, SelectionPolicy
from .policies import (
    Clock,
    SystemClock,
    cluster_of,
    grade_to_rating,
    is_mastered,
    next_mastery,
    parse_edge_id,
)


class CurriculumApplicationService(CurriculumService):
    def __init__(
        self,
        *,
        concepts: ConceptIndexRepository,
        edges: EdgeRepository,
        questions: QuestionRepository,
        states: LearnerStateRepository,
        reviews: ReviewLogRepository,
        profiles: CourseProfileRepository,
        content: ContentRepository,
        scheduler: SchedulingStrategy,
        selection: SelectionPolicy,
        propagation: CreditPropagationStrategy,
        resolve_config: Callable[[CourseProfile], EngineConfig],
        telemetry: TelemetryRepository | None = None,
        clock: Clock | None = None,
        skip_threshold: int = 3,
        hard_due_retrievability: float = 0.6,
    ) -> None:
        self._concepts = concepts
        self._edges = edges
        self._questions = questions
        self._states = states
        self._reviews = reviews
        self._profiles = profiles
        self._content = content
        self._scheduler = scheduler
        self._selection = selection
        self._propagation = propagation
        self._resolve_config = resolve_config
        self._telemetry = telemetry
        self._clock = clock or SystemClock()
        self._skip_threshold = skip_threshold
        self._hard_due_r = hard_due_retrievability
        self._last_cluster: dict[str, str] = {}  # ephemeral interleaving memory, per course

    # ----------------------------------------------------------------- helpers
    def _engine(self, course: str) -> tuple[CourseProfile, EngineConfig]:
        profile = self._profiles.get(course)
        if profile is None:
            raise ConfigError(f"no course_profile for course {course!r}; run init first")
        return profile, self._resolve_config(profile)

    def _days_to_exam(self, profile: CourseProfile, now: datetime) -> int | None:
        if profile.exam_date is None:
            return None
        return (profile.exam_date - now.date()).days

    def _log(self, kind: str, course: str, at: datetime, payload: Mapping[str, Any]) -> None:
        """Append one engagement event. Telemetry is an optional layer (like
        FIRe): when no repository is wired the signal is simply not recorded.
        A wired repository's errors propagate -- the codebase never swallows."""
        if self._telemetry is None:
            return
        self._telemetry.append(EngagementEvent(kind=kind, course=course, at=at, payload=payload))

    @staticmethod
    def _never_seen(state: LearnerState | None) -> bool:
        """No retention signal yet -- same rule as snapshot.unlock_proximity."""
        return state is None or state.stability is None

    def _gating_prereqs(self, concept_id: str) -> list[Edge]:
        """The incoming PREREQUISITE edges that actually gate unlocking.

        Only human-vetted edges gate: spine chains and manual curation. LLM-
        inferred prerequisites are useful cross-links for connection-based
        questioning, but trusting them as hard gates can close all entry points
        into a densely connected graph -- a single inferred edge targeting every
        entry concept leaves nothing learnable from cold. Every unlock reading
        (the gate itself, unlocks_ready, near_unlocks, the frontier
        breakthrough bucket) must draw from this ONE filter or checkin can call
        a concept simultaneously "ready" and "N prerequisites away".
        """
        return [
            e
            for e in self._edges.in_edges(concept_id, EdgeType.PREREQUISITE)
            if e.provenance != "inferred"
        ]

    def _prereqs_satisfied(self, concept_id: str) -> bool:
        """A concept is learnable when every gating prerequisite is mastered (SOLID+)."""
        for e in self._gating_prereqs(concept_id):
            st = self._states.get(e.src)
            if st is None or not is_mastered(st.mastery):
                return False
        return True

    def _candidate(
        self,
        concept: Concept,
        mode: NextMode,
        state: LearnerState | None,
        *,
        profile: CourseProfile,
        now: datetime,
        days_to_exam: int | None,
        hard_due: bool,
        last_cluster: str | None,
        edge_id: str | None = None,
    ) -> CandidateContext:
        retr = self._scheduler.retrievability(state, now) if state is not None else None
        return CandidateContext(
            concept=concept,
            mode=mode,
            state=state,
            retrievability=retr,
            now=now,
            profile=profile,
            cluster=cluster_of(concept.id),
            visits=state.reps if state is not None else 0,
            days_to_exam=days_to_exam,
            hard_due=hard_due,
            extra={"last_cluster": last_cluster, "edge_id": edge_id},
        )

    def _build_candidates(
        self, course: str, profile: CourseProfile, now: datetime, focus: str | None = None
    ) -> list[CandidateContext]:
        days = self._days_to_exam(profile, now)
        last = self._last_cluster.get(course)
        terms = self._focus_terms(focus)
        out: list[CandidateContext] = []
        seen: set[tuple[str, NextMode]] = set()

        # 1. due reviews
        for st in self._states.due(course, now):
            c = self._concepts.get(st.concept_id)
            if c is None or not self._in_focus(c, terms):
                continue
            r = self._scheduler.retrievability(st, now)
            out.append(
                self._candidate(
                    c, NextMode.REVIEW, st, profile=profile, now=now,
                    days_to_exam=days, hard_due=r < self._hard_due_r, last_cluster=last,
                )
            )
            seen.add((c.id, NextMode.REVIEW))

        # 2. learnable fringe (never-seen concepts whose prerequisites are mastered)
        for c in self._concepts.list_by_course(course):
            if (
                self._states.get(c.id) is None
                and self._in_focus(c, terms)
                and self._prereqs_satisfied(c.id)
            ):
                out.append(
                    self._candidate(
                        c, NextMode.TEACH, None, profile=profile, now=now,
                        days_to_exam=days, hard_due=False, last_cluster=last,
                    )
                )
                seen.add((c.id, NextMode.TEACH))

        # 3. escalated skipped connections -> forced multi-hop test
        for e in self._edges.list_by_course(course):
            if e.skip_count >= self._skip_threshold and (e.src, NextMode.TEST) not in seen:
                c = self._concepts.get(e.src)
                if c is None or not self._in_focus(c, terms):
                    continue
                out.append(
                    self._candidate(
                        c, NextMode.TEST, self._states.get(e.src), profile=profile, now=now,
                        days_to_exam=days, hard_due=True, last_cluster=last, edge_id=e.id,
                    )
                )
                seen.add((e.src, NextMode.TEST))
        return out

    @staticmethod
    def _focus_terms(focus: str | None) -> list[str]:
        """Lowercase match terms parsed from a focus string (comma/space separated)."""
        if not focus:
            return []
        return [t for t in re.split(r"[,\s]+", focus.lower()) if t]

    @staticmethod
    def _in_focus(concept: Concept, terms: Sequence[str]) -> bool:
        """In scope if no focus is set, or any term is a substring of the concept
        id or one of its source-file tokens -- so 'crypto', 'cyber-03' or 'm2' all
        scope to the right material without needing a separate module schema."""
        if not terms:
            return True
        hay = concept.id.lower() + " " + " ".join(
            (sr.file or "").lower() for sr in concept.source_refs
        )
        return any(t in hay for t in terms)

    def _pick_question(self, action: NextAction, candidates: Sequence[CandidateContext]) -> str | None:
        if action.mode is NextMode.TEACH:
            return None
        edge_id = None
        for ctx in candidates:
            if ctx.concept.id == action.concept_id and ctx.mode is action.mode:
                edge_id = ctx.extra.get("edge_id")
                break
        qs: Sequence[Question] = ()
        if edge_id:
            qs = self._questions.by_edge(edge_id)
        if not qs:
            qs = self._questions.by_concept(action.concept_id)
        return qs[0].id if qs else None

    # ------------------------------------------------------------- use-cases
    def next_action(self, course: str, *, focus: str | None = None) -> NextResult:
        profile, config = self._engine(course)
        now = self._clock.now()
        candidates = self._build_candidates(course, profile, now, focus=focus)
        if not candidates:
            raise NoCandidatesAvailable(
                f"no concepts match focus {focus!r}"
                if focus
                else "nothing is due or learnable yet"
            )
        result = self._selection.select(candidates, config=config, now=now)
        qid = self._pick_question(result.chosen, candidates)
        chosen = replace(result.chosen, question_id=qid) if qid else result.chosen
        self._last_cluster[course] = cluster_of(chosen.concept_id)
        return NextResult(chosen=chosen, candidates=result.candidates, temperature=result.temperature)

    def explain(self, concept_id: str) -> ConceptContent:
        c = self._content.get_concept_content(concept_id)
        if c is None:
            raise ContentNotFound(f"no content for concept {concept_id!r}")
        return c

    def quiz(self, concept_id: str, *, difficulty: int | None = None) -> tuple[Question, QuestionContent]:
        qs = self._questions.by_concept(concept_id, difficulty=difficulty)
        if not qs:
            raise QuestionNotFound(f"no question for concept {concept_id!r}")
        q = qs[0]
        qc = self._content.get_question_content(q.id)
        if qc is None:
            raise ContentNotFound(f"no question content for {q.id!r}")
        return q, qc

    def grade(
        self,
        *,
        concept_id: str,
        score: int,
        question_id: str | None = None,
        predicted: int | None = None,
        traversed_edges: tuple[str, ...] = (),
        skipped_edges: tuple[str, ...] = (),
    ) -> Mapping[str, Any]:
        concept = self._concepts.get(concept_id)
        if concept is None:
            raise ConceptNotFound(concept_id)
        _, config = self._engine(concept.course)
        now = self._clock.now()
        rating = grade_to_rating(score)

        # 1. primary memory update + mastery progression.
        #    The scheduler owns memory math but not identity: on a first review
        #    (prior is None) it cannot know the concept_id, so the service stamps
        #    it back on. Mastery progression is the service's job, not the
        #    scheduler's (which keeps the incoming mastery untouched).
        prior = self._states.get(concept_id)
        prior_stability = prior.stability if prior is not None and prior.stability is not None else 0.0
        recent = [e.grade for e in self._reviews.by_concept(concept_id) if e.grade is not None] + [score]
        new_state = self._scheduler.review(prior, rating, now, target_retention=config.target_retention)
        new_state = replace(new_state, concept_id=concept_id, mastery=next_mastery(score, recent))
        self._states.upsert(new_state)
        # Ripple: importance-weighted stability change, accumulated across the
        # primary concept and (below) every FIRe-credited concept, using the
        # same weighting convention as snapshot.stability_days.
        stability_gained = concept.importance * ((new_state.stability or 0.0) - prior_stability)
        self._reviews.append(
            ReviewEvent(
                concept_id=concept_id, grade=score, fsrs_rating=rating, at=now,
                question_id=question_id, predicted=predicted, scheduler_ver=self._scheduler.version,
            )
        )

        # 2. connection-skip accounting (the per-edge tracking)
        escalated: list[str] = []
        for eid in traversed_edges:
            s, t, d = parse_edge_id(eid)
            self._edges.record_exposure(s, d, t, skipped=False, at=now)
        for eid in skipped_edges:
            s, t, d = parse_edge_id(eid)
            self._edges.record_exposure(s, d, t, skipped=True, at=now)
            e = self._edges.get(s, d, t)
            if e is not None and e.skip_count >= self._skip_threshold:
                escalated.append(eid)

        # 3. FIRe: implicit credit/penalty to related concepts (experimental layer)
        fire_credits: list[dict[str, Any]] = []
        if config.enable_fire:
            event = GradeRecorded(
                concept_id=concept_id, grade=score, rating=rating, at=now,
                question_id=question_id, traversed_edges=traversed_edges, skipped_edges=skipped_edges,
            )
            for cid, irating in self._propagation.propagate(event):
                st = self._states.get(cid)
                st_stability = st.stability if st is not None and st.stability is not None else 0.0
                updated = self._scheduler.review(
                    st, irating, now, target_retention=config.target_retention
                )
                self._states.upsert(replace(updated, concept_id=cid))
                fire_credits.append({"concept_id": cid, "rating": int(irating)})
                credited = self._concepts.get(cid)
                if credited is not None:  # no concept -> no honest weight (as in snapshot)
                    stability_gained += credited.importance * ((updated.stability or 0.0) - st_stability)

        return {
            "concept_id": concept_id,
            "rating": int(rating),
            "mastery": new_state.mastery.value,
            "due_at": new_state.due_at.isoformat() if new_state.due_at else None,
            "stability": new_state.stability,
            "fire_credits": fire_credits,
            "escalated_connections": escalated,
            "ripple": {
                "count": len(fire_credits),
                "stability_days_gained": stability_gained,
            },
        }

    @staticmethod
    def _mastery_counts(
        concepts: Sequence[Concept], states: Mapping[str, LearnerState]
    ) -> dict[str, int]:
        counts = {m.value: 0 for m in Mastery}
        for c in concepts:
            st = states.get(c.id)
            counts[(st.mastery if st else Mastery.NEW).value] += 1
        return counts

    def state(self, course: str) -> Mapping[str, Any]:
        now = self._clock.now()
        concepts = self._concepts.list_by_course(course)
        states = {s.concept_id: s for s in self._states.all_for_course(course)}
        counts = self._mastery_counts(concepts, states)
        due_now = sum(1 for s in self._states.due(course, now))
        topics: dict[str, int] = {}
        for c in concepts:
            tok = c.source_refs[0].file if c.source_refs else "(none)"
            topics[tok] = topics.get(tok, 0) + 1
        return {
            "course": course,
            "total": len(concepts),
            "by_mastery": counts,
            "due_now": due_now,
            "topics": dict(sorted(topics.items())),
        }

    def checkin(self, course: str) -> Mapping[str, Any]:
        """The honest game-state reading: every number is minted by the
        snapshot module from current state; the only history consulted is the
        last "check" event, whose recorded total anchors the delta."""
        now = self._clock.now()
        concepts = list(self._concepts.list_by_course(course))
        concept_map = {c.id: c for c in concepts}
        states_list = list(self._states.all_for_course(course))
        states_map = {s.concept_id: s for s in states_list}

        stability = snapshot.stability_days(states_list, concept_map)
        last_check = self._telemetry.last("check", course) if self._telemetry else None
        delta: float | None = None
        if last_check is not None and "stability_days" in last_check.payload:
            delta = stability - float(last_check.payload["stability_days"])

        consolidation = snapshot.consolidation_report(
            states_list, last_check.at if last_check else None, now
        )
        ripe = snapshot.ripeness(states_list, now)
        prereq_in = {c.id: self._gating_prereqs(c.id) for c in concepts}
        near_unlocks = snapshot.unlock_proximity(concepts, states_map, prereq_in)
        unlocks_ready = sorted(
            c.id
            for c in concepts
            if self._never_seen(states_map.get(c.id)) and self._prereqs_satisfied(c.id)
        )

        self._log(
            "check",
            course,
            now,
            {
                "stability_days": stability,
                "holding": consolidation["holding"],
                "reviewed_since": consolidation["reviewed_since"],
            },
        )
        return {
            "course": course,
            "stability_days": stability,
            "delta_since_last_check": delta,
            "consolidation": consolidation,
            "ripeness": dict(ripe),
            "unlocks_ready": unlocks_ready,
            "near_unlocks": near_unlocks,
            "by_mastery": self._mastery_counts(concepts, states_map),
        }

    def frontier(self, course: str, *, focus: str | None = None) -> Mapping[str, Any]:
        """Up to three strategy buckets over the same candidate pool next_action
        sees. A pure read: no candidate is consumed, no question is attached,
        the interleaving memory is untouched and no RNG is drawn (the sampling
        lottery belongs to next_action) -- plus one "escalate" event."""
        profile, _config = self._engine(course)
        now = self._clock.now()
        candidates = self._build_candidates(course, profile, now, focus=focus)
        buckets: dict[str, dict[str, Any]] = {}

        teach = [c for c in candidates if c.mode is NextMode.TEACH]
        if teach:
            best = min(teach, key=lambda c: (-c.concept.importance, c.concept.id))
            buckets["push"] = {
                "concept_id": best.concept.id,
                "mode": NextMode.TEACH.value,
                "reason": (
                    "learnable now: all prerequisites satisfied "
                    f"(importance {best.concept.importance:.2f})"
                ),
                "score": float(best.concept.importance),
            }

        reviews = [c for c in candidates if c.mode is NextMode.REVIEW]
        if reviews:
            def recall(c: CandidateContext) -> float:
                return c.retrievability if c.retrievability is not None else 0.0

            best = min(reviews, key=lambda c: (recall(c), c.concept.id))
            buckets["reinforce"] = {
                "concept_id": best.concept.id,
                "mode": NextMode.REVIEW.value,
                "reason": f"review ready: retrievability {recall(best):.2f}",
                "score": float(1.0 - recall(best)),
            }

        terms = self._focus_terms(focus)
        scoped = [c for c in self._concepts.list_by_course(course) if self._in_focus(c, terms)]
        states_map = {s.concept_id: s for s in self._states.all_for_course(course)}
        prereq_in = {c.id: self._gating_prereqs(c.id) for c in scoped}
        near = snapshot.unlock_proximity(scoped, states_map, prereq_in)
        if near:
            row = min(near, key=lambda r: (not r["one_away"], r["missing"], r["concept_id"]))
            buckets["breakthrough"] = {
                "concept_id": row["concept_id"],
                "mode": NextMode.TEACH.value,
                "reason": f"{row['missing']} prerequisite(s) left to unlock",
                "score": float(1.0 / row["missing"]),
            }

        self._log(
            "escalate",
            course,
            now,
            {
                "focus": focus,
                "buckets": {name: entry["concept_id"] for name, entry in buckets.items()},
            },
        )
        return buckets

    def flag_question(self, question_id: str, *, reason: str = "") -> Mapping[str, Any]:
        """The kill switch: retire the question so it is never served again."""
        question = self._questions.get(question_id)
        if question is None:
            raise QuestionNotFound(f"no question with id {question_id!r}")
        self._questions.retire(question_id)
        concept = self._concepts.get(question.concept_id)
        course = concept.course if concept is not None else ""
        self._log(
            "item_flag",
            course,
            self._clock.now(),
            {"question_id": question_id, "concept_id": question.concept_id, "reason": reason},
        )
        return {"question_id": question_id, "status": "retired"}

    def list_courses(self) -> list[str]:
        """Return every distinct course name known to the engine, sorted.

        This is the discovery call: a tutor or agent calls it on startup to
        learn which courses exist in the graph without needing a hard-coded
        list, so a newly ingested course (e.g. GPUKernelOptimization) is
        visible immediately.
        """
        return list(self._concepts.list_courses())

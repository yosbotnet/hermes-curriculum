# Motivation Layer ("Homestead" + "Research Frontier") Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Project the engine's real learning state into game-shaped, gain-framed feedback (30-second check-in, strategy frontier, grade ripple report), with spine-based trusted prerequisite edges for the upcoming GPU kernel optimization corpus.

**Architecture:** The engine mints every number deterministically (new pure module `engine/snapshot.py` + widened `CurriculumService` use-cases); the LLM (Hermes) only narrates payloads it receives over MCP. New telemetry storage records every check/escalation/session so the experiment is measurable from day one. Ingestion gains edge provenance: spine sources produce trusted PREREQUISITE chains; LLM-inferred edges carry capped confidence and are auditable.

**Tech Stack:** Python 3.11+, stdlib-only engine (repo rule), pytest, Postgres+pgvector adapter behind ports, in-memory adapters for tests.

## Global Constraints

- Determinism: no clock reads or RNG inside engine/service logic; time comes from the injected `Clock`, randomness from injected `random.Random` (repo rule, see selection.py).
- Engine modules (`engine/`) are pure: no I/O, standard library only.
- All new domain entities frozen slotted dataclasses, updated via `dataclasses.replace`.
- Gain-framing vocabulary rule: payload keys and CLI copy never use "overdue", "debt", "behind", "late". Ripe items are "ready". Granularity of ripeness is DAYS, never hours.
- No emojis and no special characters in any script, code, or CLI output (user rule).
- Numbers in payloads are facts about what happened; never phrase contingent rewards.
- Tests must pass with in-memory adapters only: `make test` needs no DB, no API key.
- Frequent commits; conventional-commit style messages matching repo history.

## Execution Waves (parallelization map)

- Wave 1 (parallel): Task 1 [opus], Task 2 [fable], Task 3 [sonnet]
- Wave 2 (parallel, after Task 1): Task 4 [sonnet], Task 5 [opus]
- Wave 3 (after Tasks 1, 2, 4): Task 6 [fable]
- Wave 4 (parallel, after Task 6): Task 7 [sonnet], Task 8 [opus]
- Deferred to v2 (do NOT build now): session-end unlock bias in selection, divergence alarm, prestige learning-rate multipliers, push notifications.

---

### Task 1: Domain + schema + ports foundation

**Files:**
- Modify: `src/curriculum/domain/entities.py` (Edge, Question)
- Create: `schema/002_motivation.sql`
- Modify: `src/curriculum/ports/repositories.py` (QuestionRepository.retire, new TelemetryRepository)
- Create: `src/curriculum/domain/telemetry.py`
- Test: `tests/test_telemetry_domain.py`

**Interfaces (Produces):**
- `Edge` gains fields `provenance: str = "inferred"` (one of "spine" | "inferred" | "manual") and `confidence: float = 0.6`.
- `Question` gains field `status: str = "active"` ("active" | "retired").
- New frozen dataclass in `domain/telemetry.py`:

```python
@dataclass(frozen=True, slots=True)
class EngagementEvent:
    kind: str                 # "check" | "escalate" | "session_start" | "session_end" | "item_flag"
    course: str
    at: datetime
    payload: Mapping[str, Any] = field(default_factory=dict)
```

- New port in `ports/repositories.py`:

```python
class TelemetryRepository(ABC):
    @abstractmethod
    def append(self, event: EngagementEvent) -> None: ...
    @abstractmethod
    def last(self, kind: str, course: str) -> EngagementEvent | None: ...
    @abstractmethod
    def list_by_course(self, course: str) -> Sequence[EngagementEvent]: ...
```

- `QuestionRepository` gains `@abstractmethod def retire(self, question_id: str) -> None: ...` and the docstring contract that `by_concept`/`by_edge` MUST exclude questions with `status == "retired"`.

**Steps:**

- [ ] **Step 1: Write failing tests** in `tests/test_telemetry_domain.py`: construct `EngagementEvent`, assert frozen (assigning raises `FrozenInstanceError`); construct `Edge(src="a", dst="b", type=EdgeType.PREREQUISITE)` and assert `edge.provenance == "inferred"` and `edge.confidence == 0.6`; construct `Question(id="q", concept_id="c")` and assert `q.status == "active"`.
- [ ] **Step 2:** Run `pytest tests/test_telemetry_domain.py -v`, expect FAIL (missing module/fields).
- [ ] **Step 3:** Implement entity fields (append after existing fields to keep positional compat), `domain/telemetry.py`, and the port additions.
- [ ] **Step 4:** Write `schema/002_motivation.sql`:

```sql
-- Motivation layer: telemetry, question kill switch, edge provenance.
ALTER TABLE question ADD COLUMN IF NOT EXISTS status text NOT NULL DEFAULT 'active';
ALTER TABLE edge ADD COLUMN IF NOT EXISTS provenance text NOT NULL DEFAULT 'inferred'
    CHECK (provenance IN ('spine', 'inferred', 'manual'));
ALTER TABLE edge ADD COLUMN IF NOT EXISTS confidence real NOT NULL DEFAULT 0.6;
CREATE TABLE IF NOT EXISTS engagement_log (
    id      bigserial PRIMARY KEY,
    kind    text NOT NULL CHECK (kind IN ('check', 'escalate', 'session_start', 'session_end', 'item_flag')),
    course  text NOT NULL,
    at      timestamptz NOT NULL DEFAULT now(),
    payload jsonb NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS engagement_course_idx ON engagement_log (course, kind, at DESC);
```

- [ ] **Step 5:** Run full `pytest tests -q`; fix any adapter constructor breakage caused by new fields (defaults should prevent it). Expect PASS.
- [ ] **Step 6:** Commit: `feat: add telemetry domain, edge provenance, question status (schema 002)`

### Task 2: Pure snapshot metrics module

**Files:**
- Create: `src/curriculum/engine/snapshot.py`
- Test: `tests/test_snapshot.py`

**Interfaces (Consumes):** existing entities only. **(Produces)** pure functions used by Task 6:

```python
def stability_days(states: Sequence[LearnerState], concepts: Mapping[str, Concept]) -> float
    # sum over states with stability is not None of concept.importance * state.stability
def ripeness(states: Sequence[LearnerState], now: datetime) -> Mapping[str, list[str]]
    # buckets by due_at DAY granularity, gain-framed keys:
    # {"ready_now": [...], "ready_tomorrow": [...], "ready_this_week": [...], "holding": [...]}
    # ready_now = due_at date <= now date (includes past-due: NEVER a separate "overdue" bucket)
def unlock_proximity(course_concepts: Sequence[Concept], states: Mapping[str, LearnerState],
                     prereq_in_edges: Mapping[str, Sequence[Edge]]) -> list[dict]
    # for each never-seen concept with at least one unmastered prereq:
    # {"concept_id", "missing": <count of unmastered prereqs>, "one_away": bool}
    # one_away = exactly one prereq unmastered AND that prereq's mastery is Mastery.LEARNING or better.
    # sorted by (missing asc, concept_id asc); mastered = mastery in (SOLID, EXAM_READY)
def consolidation_report(states: Sequence[LearnerState], since: datetime | None, now: datetime) -> dict
    # {"holding": <count of states with due_at > now>,  # intervals still intact = knowledge held
    #  "reviewed_since": <count with last_review >= since>} ; since None -> reviewed_since 0
```

**Steps:**

- [ ] **Step 1: Write failing tests** in `tests/test_snapshot.py` covering: importance weighting in `stability_days` (two states, importances 1.0 and 0.5, stabilities 10 and 20 -> 20.0); never-seen states (stability None) excluded; ripeness bucket edges (due yesterday and due today both in `ready_now`; due tomorrow in `ready_tomorrow`; due in 3 days in `ready_this_week`; due in 30 days in `holding`); `unlock_proximity` one_away logic with a two-prereq concept where one prereq is SOLID and one is LEARNING (missing == 1, one_away True) and where the missing prereq is NEW (one_away False); determinism (same inputs -> identical output, no clock reads).
- [ ] **Step 2:** Run `pytest tests/test_snapshot.py -v`, expect FAIL.
- [ ] **Step 3:** Implement `snapshot.py` in the style of `engine/scoring.py`: module docstring explaining the honest-currency rules (importance weighting defeats corpus padding; day granularity because retrievability moves on day scale, hour precision would be fake urgency; no debt-shaped buckets by design), named constants, pure functions, stdlib only.
- [ ] **Step 4:** Run tests, expect PASS. Run `pytest tests -q` for regressions.
- [ ] **Step 5:** Commit: `feat: pure snapshot metrics (stability capital, ripeness, unlock proximity)`

### Task 3: GPU corpus guide

**Files:**
- Create: `docs/corpus-gpu-kernels.md`

**Steps:**

- [ ] **Step 1:** Write the guide with sections: (a) Spine + satellites model and why spine ordering becomes trusted prerequisite edges; (b) Phase 1 CUDA foundations + performance model (PMPP 4th ed chapter list as the spine, or free alternative: GPU MODE lectures + CUDA C++ Programming Guide section order; satellites: CUDA Best Practices Guide, Nsight Compute docs); (c) Phase 2 optimization craft (Simon Boehm CUDA matmul worklog steps, Horace He "Making Deep Learning Go Brrr", tensor cores, CUTLASS/CuTe docs, Triton tutorials, Citadel Volta/Turing microbenchmarking papers); (d) Phase 3 inference systems (vLLM paper + docs, PagedAttention, FlashAttention 1/2/3, continuous batching; vLLM source LAST); (e) checkpoint-concept ladder (naive matmul -> coalesced -> tiled -> double-buffered -> tensor-core -> flash-attention forward) each phrased as a concept whose mastery the learner self-gates on having built the artifact; (f) a concrete `corpus.json` example using the `spine: true` flag from Task 5 with plain-text file paths and the note that the user supplies their own legally obtained materials, none are committed. Plain prose, no emojis.
- [ ] **Step 2:** Commit: `docs: GPU kernel optimization corpus guide (spine+satellites, phased)`

### Task 4: Storage adapters for new ports

**Files:**
- Modify: `src/curriculum/storage/memory.py` (InMemory question retire/filtering, new InMemoryTelemetryRepository)
- Modify: `src/curriculum/storage/postgres.py` (question status column + retire, edge provenance/confidence columns in upsert/row mapping, PostgresTelemetryRepository over engagement_log)
- Test: `tests/test_telemetry_store.py`, extend `tests/test_memory.py` follow existing patterns; postgres tests follow the existing `tests/test_postgres.py` skip-without-DB pattern.

**Interfaces (Consumes):** Task 1 ports/entities exactly as specified there.

**Steps:**

- [ ] **Step 1: Write failing tests**: in-memory telemetry append/last/list ordering (append three events, `last("check", course)` returns the newest by `at`); retired question excluded from `by_concept` and `by_edge` but still returned by `get`; edge round-trips preserve provenance/confidence.
- [ ] **Step 2:** Run new tests, expect FAIL.
- [ ] **Step 3:** Implement in-memory adapters; then mirror in postgres.py (SQL includes new columns; telemetry insert + `ORDER BY at DESC LIMIT 1` for last).
- [ ] **Step 4:** Run `pytest tests -q` (postgres tests skip without DB). Expect PASS.
- [ ] **Step 5:** Commit: `feat: telemetry + retire + provenance in memory and postgres adapters`

### Task 5: Spine ingestion + edge audit

**Files:**
- Read first: `src/curriculum/ingestion/passes.py`, `src/curriculum/ingestion/pipeline.py`, `src/curriculum/app/build.py` (corpus.json parsing)
- Modify: `src/curriculum/ingestion/passes.py` (new SpinePass; InferEdgesPass stamps provenance="inferred" and caps confidence at 0.85, never overwriting an existing spine edge)
- Modify: `src/curriculum/ingestion/pipeline.py` (SpinePass ordered after DedupePass, before InferEdgesPass)
- Modify: `src/curriculum/app/build.py` (per-source `"spine": true` flag from corpus.json reaches the pass)
- Test: `tests/test_spine.py`

**Interfaces (Produces):** SpinePass behavior contract: for concepts extracted from a source marked spine, ordered by (source file order in corpus, then first source_ref line), emit `Edge(src=prev, dst=next, type=EdgeType.PREREQUISITE, provenance="spine", confidence=1.0, rationale="spine order: <source name>")` chaining consecutive concepts. InferEdgesPass must skip creating any edge (src, dst, PREREQUISITE) where a spine edge already exists.

**Steps:**

- [ ] **Step 1:** Read the three files above; identify IngestionContext fields carrying source identity/order for extracted concepts. If concept-to-source attribution is not available on the context, extend IngestionContext minimally (e.g. a `source_of: dict[str, tuple[str, int]]` mapping concept id to (source name, order index)) and populate it in ExtractPass.
- [ ] **Step 2: Write failing tests** in `tests/test_spine.py` using the existing FakeLlm/in-memory pipeline fixtures from `tests/test_ingestion.py` as the template: a corpus with one spine source yielding concepts A, B, C in order produces exactly spine edges A->B->C (provenance "spine", confidence 1.0); a non-spine source produces none; an InferEdgesPass proposal duplicating A->B does not overwrite the spine edge; inferred edges carry provenance "inferred" and confidence <= 0.85.
- [ ] **Step 3:** Run, expect FAIL. Implement. Run, expect PASS; full suite for regressions.
- [ ] **Step 4:** Commit: `feat: spine-trusted prerequisite chains and capped-confidence inferred edges`

### Task 6: Service use-cases (checkin, frontier, ripple, flag)

**Files:**
- Modify: `src/curriculum/application/service.py`
- Modify: `src/curriculum/ports/service.py` (add the four methods to the CurriculumService ABC)
- Modify: `src/curriculum/application/composition.py` (inject TelemetryRepository)
- Test: `tests/test_motivation_service.py`

**Interfaces (Consumes):** Task 1 `TelemetryRepository`/`EngagementEvent`, Task 2 snapshot functions, Task 4 adapters. **(Produces)** on `CurriculumService`:

```python
def checkin(self, course: str) -> Mapping[str, Any]
    # computes via snapshot module; logs EngagementEvent(kind="check", payload=totals);
    # returns {"course", "stability_days": float, "delta_since_last_check": float | None,
    #          "consolidation": {...}, "ripeness": {...}, "unlocks_ready": [concept ids
    #          never seen with all prereqs satisfied], "near_unlocks": [...],
    #          "by_mastery": {...}}
    # delta = stability_days minus the value stored in the last "check" event payload (None on first check).
def frontier(self, course: str, *, focus: str | None = None) -> Mapping[str, Any]
    # builds candidates exactly as next_action does, then returns up to three strategy buckets:
    # {"push": best TEACH candidate, "reinforce": best REVIEW candidate (lowest retrievability
    #  among top-scored), "breakthrough": best near-unlock from snapshot.unlock_proximity
    #  (one_away first)} each as {"concept_id", "mode", "reason", "score"}; omit empty buckets;
    # logs kind="escalate". Does NOT advance _last_cluster (choosing happens later via next/quiz).
def flag_question(self, question_id: str, *, reason: str = "") -> Mapping[str, Any]
    # retires the question, logs kind="item_flag" with reason; returns {"question_id", "status": "retired"}
# grade(...) return dict gains "ripple": {"count": len(fire_credits),
#   "stability_days_gained": sum of (new - old) stability across the primary concept and all
#   fire-credited concepts, importance-weighted using snapshot.stability_days conventions}
```

**Steps:**

- [ ] **Step 1: Write failing tests** in `tests/test_motivation_service.py` using the existing service-test fixtures (in-memory repos, fake scheduler pattern from current tests): first `checkin` has `delta_since_last_check is None` and logs one "check" event; second checkin after a graded review reports a positive delta; `frontier` on a course with one due review, one learnable concept, and one near-unlock returns all three buckets with distinct concept ids; `flag_question` makes the question disappear from `quiz()` (expect `QuestionNotFound` when it was the only one) and logs "item_flag"; `grade` return includes `ripple.stability_days_gained > 0` when FIRe credits fire.
- [ ] **Step 2:** Run, expect FAIL. Implement: capture prior stabilities before scheduler updates in `grade` step 1 and step 3 to compute the ripple sum; wire telemetry repo through `__init__` and `composition.py` (default to the in-memory implementation when not provided, keeping existing constructions working).
- [ ] **Step 3:** Run new tests then the full suite, expect PASS.
- [ ] **Step 4:** Commit: `feat: checkin, frontier, question flagging, grade ripple report`

### Task 7: CLI check + flag commands

**Files:**
- Modify: `src/curriculum/cli.py`
- Test: extend `tests/test_cli.py` following its existing command-test pattern

**Interfaces (Consumes):** Task 6 `checkin`, `flag_question`.

**Steps:**

- [ ] **Step 1: Write failing tests**: `curriculum check --course X` exits 0 and prints the compact human report (assert key lines present: "Knowledge held:", "Ready today:", "Unlocked:", no word "overdue" anywhere); `curriculum check --course X --json` emits the raw payload via `_emit`; `curriculum flag-question <id> --reason "ambiguous"` exits 0.
- [ ] **Step 2:** Run, expect FAIL. Implement handlers following the existing deferred-import pattern; human rendering is a small pure `_render_check(payload) -> str` function (testable directly), max ~12 lines of output, plain ASCII.
- [ ] **Step 3:** Run tests, expect PASS. Commit: `feat: curriculum check and flag-question commands`

### Task 8: MCP tools + narration contract

**Files:**
- Modify: `src/curriculum/mcp/server.py` (register tools `checkin`, `frontier`, `flag_question`; extend grade serialization with ripple)
- Create: `docs/narration-contract.md`
- Test: extend `tests/test_mcp.py` using its existing pure `_call_*` router test pattern (no mcp SDK needed)

**Interfaces (Consumes):** Task 6 service methods; payload shapes exactly as Task 6 defines them.

**Steps:**

- [ ] **Step 1: Write failing tests** for the new `_call_checkin`, `_call_frontier`, `_call_flag_question` routers returning JSON-able dicts.
- [ ] **Step 2:** Run, expect FAIL. Implement thin routers mirroring existing tools.
- [ ] **Step 3:** Write `docs/narration-contract.md` — the host-prompt contract: engine mints, LLM narrates; numbers verbatim from payloads, never invented or improved; informational framing only (describe what happened, never offer contingent rewards); gain-framed vocabulary (ready, held, unlocked; never overdue/debt/behind); length budgets (check narration max 3 sentences; ripple 1 sentence; frontier max 2 sentences per option); the cliffhanger rule (LLM crafts an almost-answerable question ONLY from the engine-chosen unlock concept and its source_refs); no emojis. Include a verbatim system-prompt block Hermes can paste.
- [ ] **Step 4:** Run full suite, expect PASS. Commit: `feat: MCP checkin/frontier/flag tools and the narration contract`

---

## Self-review notes

- Spec coverage: check-in loop (T2+T6+T7), frontier choice (T6+T8), ripple (T6), honest currency (importance weighting T2, day granularity T2, gain-framing everywhere), telemetry-first (T1+T4+T6), item kill switch (T1+T4+T6+T7), edge provenance + spine + audit path (T1+T5), GPU corpus (T3+T5). Deferred list is explicit.
- Type consistency: EngagementEvent/TelemetryRepository names match across T1/T4/T6; snapshot signatures match T2/T6; payload key names match T6/T7/T8.
- Placeholders: none; Task 5 Step 1 is a deliberate read-first step because IngestionContext internals must not be guessed.

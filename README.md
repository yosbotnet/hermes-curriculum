# Curriculum engine

A knowledge-graph engine that decides **what to teach or test next, and in what order**,
so a tutor LLM never has to make curriculum decisions mid-session. The model becomes the
*delivery* layer; this is the *curriculum* layer. You point it at your own plain-text
course materials; it builds a concept/edge graph, generates grounded exam questions, and
serves the result to a host (Hermes) over MCP.

No source materials and no secrets are committed here -- you supply your own materials via
`corpus.json` and your `CURRICULUM_API_KEY` at runtime. Inference goes through any
OpenAI-compatible endpoint (Nous, NVIDIA NIM, vLLM, ...), selected with `CURRICULUM_BASE_URL`.

## Quickstart

Full bootstrap (Docker + database, install, build, serve) lives in **[AGENTS.md](AGENTS.md)**.
One screen:

```sh
docker compose up -d db                                   # Postgres+pgvector on :5433
uv venv && uv pip install -e '.[postgres,mcp,nous]'       # install with all adapters
export CURRICULUM_API_KEY=...                              # the only required setting
# export CURRICULUM_BASE_URL=https://integrate.api.nvidia.com/v1   # any OpenAI-compatible endpoint
cp corpus.example.json corpus.json                         # then edit it for your course
.venv/bin/curriculum build corpus.json                     # ingest -> link -> questions
.venv/bin/curriculum status --course <your-course>         # read-only graph counts
```

`CURRICULUM_API_KEY` / `CURRICULUM_BASE_URL` are the primary names; the legacy `NOUS_API_KEY`
/ `NOUS_BASE_URL` still work as fallbacks (the generic names win when both are set).

Run `.venv/bin/curriculum doctor` first to check prerequisites (docker, DB, key, bundle).
Tests need neither a database nor a key: `make test`.

## Architecture (ports & adapters / hexagonal)

```
domain/        immutable entities + DTOs + events (no I/O, no deps)
ports/         abstract interfaces: repositories, strategies, providers, service
engine/        FSRS scheduler, FIRe propagation, scoring terms, selection policy
storage/       InMemory + Postgres repositories, OKF content repository
okf/           OKF v0.1 (de)serialization
ingestion/     multipass pipeline: extract -> dedupe -> spine -> infer-edges -> verify
linking/       embedding-guided edge repair for isolated concepts
archetypes/    course strategy templates (conceptual-written, procedural, ...)
sync/          OKF <-> Postgres reconciliation (hash-keyed, one-way)
application/    the CurriculumService use-cases (next/explain/quiz/grade/state)
app/           config-driven build orchestration (ingest/link/questions/status)
cli/           the agent-facing `curriculum` entrypoint
mcp/           stdio MCP server exposing the service to Hermes
```

The core depends only on `domain` + `ports`; everything external is a swappable adapter
(Dependency Inversion). The engine therefore runs and tests with in-memory + fake adapters,
with no Postgres and no paid inference. Heavy/optional imports (`psycopg`, `mcp`, the
OpenAI-compatible HTTP provider) are deferred into the functions that use them, so the
package imports cleanly on a
machine that has none of them installed.

## Storage split (polyglot, single-ownership)

- **OKF markdown bundle** owns the *content*: concept/question prose and citations.
  Human-readable, git-diffable, hand-editable; the source of truth for text.
- **Postgres + pgvector** owns the *structure + metadata + state*: edges, importance,
  skip-counts, embeddings, and all per-learner FSRS/review state.
- A one-way, content-hash-keyed **sync** reconciles OKF -> Postgres (and re-embeds changed
  content), so no fact lives authoritatively in two places and nothing drifts.

## The engine

- **FSRS** -- the spaced-repetition scheduler (when each concept is next due).
- **FIRe** -- importance propagation across the edge graph after a graded answer.
- **Selection** -- a weighted-sampling policy over five scoring terms (urgency, difficulty
  fit, exploration, interleave penalty, coverage) that picks the single best next action and
  exposes the full ranked field.

These feed the eight MCP tools -- `next`, `explain`, `quiz`, `grade`, `state`, plus the
motivation-layer `checkin`, `frontier`, and `flag_question` -- which are thin adapters over
the same `CurriculumService` use-cases the unit tests drive directly. The engine mints every
number deterministically; the tutor LLM only narrates them (see
[`docs/narration-contract.md`](docs/narration-contract.md)). `curriculum check` renders the
same check-in payload in the terminal with no LLM involved.

## Design docs

- [`docs/okf-spec.md`](docs/okf-spec.md) -- the OKF v0.1 content-bundle format (frontmatter,
  ids, citations) that the file content repository reads and writes.

## Build & serve

See **[AGENTS.md](AGENTS.md)** for the full runbook (prerequisites, env vars, reset, and
troubleshooting). The `Makefile` wraps the common steps: `make db`, `make install`,
`make build`, `make link`, `make questions`, `make status`, `make mcp`, `make reset`,
`make test`.

# Issue #1 -- Generalize inference provider to OpenAI-compatible APIs

## Summary
The provider already spoke OpenAI-compatible HTTP; this change makes the *config
and naming* vendor-neutral. Generic `CURRICULUM_API_KEY` / `CURRICULUM_BASE_URL`
are now primary, the legacy `NOUS_*` names remain as backward-compatible
fallbacks (generic wins when both are set), the concrete provider moved to
`providers_openai_compatible.py`, and `providers_nous.py` is now a thin alias
shim. Every consumer (config, build stages, CLI doctor, MCP registration) and
the docs were updated coherently.

## What changed (files)
- `src/curriculum/config.py` -- renamed Settings fields `nous_api_key`/`nous_base_url`
  -> `api_key`/`base_url`; `load()` reads `CURRICULUM_API_KEY`/`CURRICULUM_BASE_URL`
  first, falling back to `NOUS_API_KEY`/`NOUS_BASE_URL`, then the default. The
  vendor default base URL is unchanged (keeps existing Nous setups working).
- `src/curriculum/providers_openai_compatible.py` -- NEW. The moved implementation:
  `OpenAICompatibleLlm` (was `NousLlm`) and `OpenAICompatibleEmbedder` (was
  `NousEmbedder`). Request shape byte-for-byte identical; only the stderr log
  prefixes changed (`[NousLlm]` -> `[OpenAICompatibleLlm]`).
- `src/curriculum/providers_nous.py` -- now a thin compatibility shim:
  `NousLlm = OpenAICompatibleLlm`, `NousEmbedder = OpenAICompatibleEmbedder`,
  `__all__` preserved. Existing imports keep working.
- `src/curriculum/app/build.py` -- imports the new module; constructs
  `OpenAICompatibleLlm`/`OpenAICompatibleEmbedder`; now passes
  `base_url=settings.base_url` at every construction site (previously
  `settings.nous_base_url` was never used -- a latent bug: `CURRICULUM_BASE_URL`
  had no effect); `settings.nous_api_key` -> `settings.api_key`;
  `_require_nous_key` -> `_require_api_key` with a generic, fallback-aware message.
- `src/curriculum/cli.py` -- `_register_argv` emits `CURRICULUM_API_KEY` +
  `CURRICULUM_BASE_URL` (added; previously the base URL was never registered);
  `_render` masks `CURRICULUM_API_KEY="$CURRICULUM_API_KEY"` (secret never
  printed); `_cmd_mcp_register` uses `settings.api_key`; `_check_nous` ->
  `_check_api_key` (label `CURRICULUM_API_KEY`, reads `settings.api_key`, green on
  either name).
- Docstring-only vendor-neutrality: `providers_fake.py`, `ports/providers.py`,
  `ingestion/pipeline.py` (x2), `ingestion/passes.py` ("Nous-backed" ->
  "OpenAI-compatible").
- `README.md`, `AGENTS.md` -- vendor-neutral setup: generic vars primary, an
  NVIDIA-NIM example (`https://integrate.api.nvidia.com/v1`), and an explicit note
  that `NOUS_*` still works as a fallback. Env-var table updated (labels, both
  `settings.api_key`/`settings.base_url`, legacy-fallback column notes).
- Tests: NEW `tests/test_config.py` (7 cases) and
  `tests/test_providers_openai_compatible.py` (9 cases); `tests/test_cli.py`
  doctor assertion updated `nous_api_key` -> `curriculum_api_key`.

## Every NOUS_ site found and its disposition
Grep target `NOUS_` / `nous` across src, tests, Makefile, README, AGENTS,
docker-compose:
- `config.py` (fields + `load`) -- renamed to generic; `NOUS_*` kept as fallback.
- `providers_nous.py` -- converted to shim (module name kept for import compat).
- `app/build.py` (import, 5 constructions, `_require_nous_key`, docstrings) --
  all updated to generic settings + provider names; base_url now threaded through.
- `cli.py` (`_register_argv`, `_render`, `_cmd_mcp_register`, `_check_nous`,
  `_cmd_doctor`) -- all updated; MCP registration emits generic names, no secrets.
- `tests/test_cli.py:115` -- assertion updated to the new doctor label.
- `README.md`, `AGENTS.md` -- rewritten vendor-neutral (see above).
- `providers_fake.py`, `ports/providers.py`, `ingestion/passes.py`,
  `ingestion/pipeline.py` -- "Nous-backed" docstrings genericised.
- `docker-compose.yml` -- NO `NOUS_` references (only the Postgres service).
- `Makefile` / `pyproject.toml` -- the `.[nous]` optional-dependency *extra*
  (`nous = ["httpx>=0.27"]`) matches lowercase `nous`, not the `NOUS_` env-var
  target. LEFT AS-IS deliberately: renaming a published extra is a breaking
  change and out of the issue's scope (env-var/provider naming). See Concerns.

## Acceptance criteria -- evidence
1. Legacy fallback: `test_config.LegacyFallbackTest` (NOUS_API_KEY/NOUS_BASE_URL
   populate api_key/base_url). Verified live: `_check_api_key(load({'NOUS_API_KEY':
   'legacykey'}))` -> `('CURRICULUM_API_KEY', True, 'set')`.
2. Generic precedence: `test_config.PrecedenceTest`. Verified live: both set ->
   `api_key == 'gen'`.
3. Provider request shape: `test_providers_openai_compatible` faking
   `urllib.request.urlopen`, asserting URL (`/chat/completions`, `/embeddings`),
   POST, `Authorization: Bearer <key>`, JSON body (model/messages/temperature/
   max_tokens; input), trailing-slash normalisation, and order-preserving embed
   merge. Implementation copied unchanged, so the shape is identical.
4. Legacy import compat: `test_providers_openai_compatible.LegacyShimImportTest`
   -- `providers_nous.NousLlm is OpenAICompatibleLlm`, `NousEmbedder` likewise,
   and both remain constructible.
5. Offline suite green with no Docker / no key: `PYTHONPATH=src python -m pytest
   tests -q` -> **446 passed, 32 skipped, 4 subtests passed** (baseline was 430
   passed / 32 skipped; +16 from the two new test modules).

## TDD RED -> GREEN evidence
- RED: `tests/test_config.py` -> 6 failed / 3 passed
  (`AttributeError: 'Settings' object has no attribute 'base_url'`);
  `tests/test_providers_openai_compatible.py` ->
  `ModuleNotFoundError: No module named 'curriculum.providers_openai_compatible'`.
- GREEN (after config + new module + shim): the two files -> 16 passed.
- Full suite after all consumer + doc changes: 446 passed, 32 skipped.

## Manual smoke tests
- `curriculum mcp-register` (with `CURRICULUM_API_KEY=supersecret123`, hermes
  absent) printed:
  `... --env CURRICULUM_API_KEY="$CURRICULUM_API_KEY"
  CURRICULUM_BASE_URL=https://inference-api.nousresearch.com/v1 ...` --
  generic names, base URL registered, secret NOT echoed.
- doctor `_check_api_key`: legacy -> OK; both set -> generic value; unset -> MISS
  with `export CURRICULUM_API_KEY` hint.

## Self-review
- Field rename `nous_api_key`/`nous_base_url` -> `api_key`/`base_url` is a public
  attribute change on `Settings`, but the only consumers are in-repo (build, cli),
  all updated; no test referenced the old field names. The issue asked for generic
  names to be "primary", and a clean rename (vs. keeping vendor-named fields) reads
  best. Not adding legacy property aliases keeps the frozen/slots dataclass simple;
  backward compat is required only at the ENV-VAR layer, which is preserved.
- `base_url` is now actually plumbed into the providers; before this change
  `CURRICULUM_BASE_URL`/`NOUS_BASE_URL` were loaded but ignored by build.py.
- Provider constructor defaults (incl. the vendor base URL) left unchanged so
  request shape is provably identical and direct instantiation is unaffected.
- No emojis / non-ASCII introduced; tests are deterministic and fully offline
  (HTTP layer faked, `load` fed explicit env dicts).

## Concerns
- The `nous` optional-dependency extra (`pyproject.toml`, and the
  `.[postgres,mcp,nous]` install line in README/AGENTS/Makefile) is left named
  `nous`. It is vestigial anyway -- it installs `httpx`, which the provider does
  NOT use (the client is stdlib `urllib`). Renaming/removing it is a separate,
  breaking packaging change I judged out of scope; flagging for a follow-up.
- The default `base_url` still points at Nous. That preserves backward
  compatibility, but a fully vendor-neutral default (or no default, forcing an
  explicit endpoint) could be considered in a follow-up.
- `doctor`'s DB probe blocks when Postgres is unreachable (pre-existing
  behaviour, unrelated to this change); noted only because it slowed manual CLI
  smoke testing.

# Issue #3 - persist curriculum build logs for failed runs

## Problem
When `curriculum build` (or its `ingest`/`link`/`questions` stages) stalls, times
out, or fails, no durable log artifact existed: diagnostics died with the terminal
session. We need one durable log file per invocation, at a predictable path,
carrying timestamps, the command/stage, the pid, per-source progress, and the FULL
provider/error traceback on failure - and it must survive a kill mid-run.

## Design
Logging lives strictly at the app/cli layer; the engine (`engine`/`domain`/`ports`)
stays pure and untouched.

- **`curriculum.config.Settings.log_dir`** (new field, default `"logs"`), loaded
  from `CURRICULUM_LOG_DIR` following the exact existing env pattern in
  `config.load`.
- **`curriculum.app.build_logging`** (new module, stdlib `logging` only):
  - `log_path_for(settings, command, *, now=None, pid=None)` -> predictable path
    `<log_dir>/build-<UTC-YYYYMMDDThhmmssZ>-<pid>.log`. `now`/`pid` are injectable
    only so the name is deterministically assertable in a test.
  - `start_build_log(settings, command)` -> `(logger, path)`. Creates `log_dir`,
    attaches a `logging.FileHandler` (`mode="w"`, `delay=False`). `FileHandler`
    inherits `StreamHandler.emit`, which flushes after every record, so nothing is
    buffered until the end: a killed process still leaves the partial log. The
    formatter stamps every line with a UTC ISO instant (`formatter.converter =
    time.gmtime`) and `pid=%(process)d`. A unique per-process logger name
    (`pid` + monotonic counter) prevents duplicate handlers across invocations.
    `propagate=False` keeps build lines out of the host's root logger.
  - `close_build_log(logger)` - idempotent flush/close/detach for a `finally`.
  - `NULL_LOGGER` - a shared no-op logger so orchestration called without a real
    log (unit tests, embedded callers) is a cheap no-op, not an `is None` branch.
- **`curriculum.app.build`**: `ingest`/`link`/`generate_questions` gained an
  optional `logger` parameter (default -> `NULL_LOGGER`), keeping every existing
  call site working. They log stage start/done and per-source progress; on failure
  they call `logger.exception(...)` which records the FULL traceback + error text.
  `ingest` still tolerates a single bad source (batch not aborted) but now logs its
  traceback instead of silently `continue`-ing. `link`/`questions` log the
  traceback then re-raise (preserving the CLI's exit-code discipline).
- **`curriculum.cli`**: `_open_build_log`/`_close_build_log` helpers (lazy-imported
  so `--help`/`doctor` stay free of the app layer). Each of `ingest`/`link`/
  `questions`/`build` opens the log first, PRINTS the path to stdout up front,
  threads the logger into the orchestration, records a terse failure line, and
  always closes in a `finally`. Doctor gains `_check_log_dir` - one OK/MISS line
  reporting the log dir and whether it is writable, side-effect free (uses
  `os.access`, never creates the dir), matching the existing probe style.
- **`.gitignore`**: added `logs/`.

## TDD - RED then GREEN
Tests written first. RED evidence (worktree src on PYTHONPATH, since the editable
install otherwise resolves to the main checkout):

```
tests/test_build_logging.py: ImportError: cannot import name 'build_logging'
  from 'curriculum.app'  -> collection error (module did not exist yet)
```

After implementing, GREEN:

```
tests/test_build_logging.py tests/test_cli.py -> 22 passed
```

Test cases covered:
- predictable filename carries UTC timestamp + pid (regex-pinned, injected
  `now`/`pid`);
- `start_build_log` creates the file at the predictable path and the header is
  readable back MID-RUN (proves per-record flush, not end-of-run buffering);
- timestamps are UTC ISO with trailing `Z`;
- `close_build_log` is idempotent and detaches handlers;
- `CURRICULUM_LOG_DIR` override respected; default is `logs`;
- a FAILING `ingest` (source path that does not exist -> raises before any Nous
  call, fully offline) leaves the partial log with `stage=ingest`, the source
  token, and `Traceback (most recent call last)`;
- CLI `ingest` prints the `build-*.log` path (under the override dir) to stdout and
  actually creates the file;
- `doctor` reports a `log dir` line.

## Full-suite result
`PYTHONPATH=<worktree>/src python -m pytest tests -q`:
**440 passed, 32 skipped, 4 subtests passed** (baseline was 430 passed / 32
skipped; +10 new tests, zero regressions).

## Manual end-to-end
`curriculum ingest` on a manifest with a missing source (temp `CURRICULUM_LOG_DIR`)
printed the log path up front and wrote a durable log containing the header
(command + pid), `stage=ingest starting`, an `ERROR` line with the FULL
`FileNotFoundError` traceback, and `stage=ingest done`. `git status` stayed clean.

## Self-review / concerns
- **Provider timeout text**: `NousLlm` swallows HTTP/timeout errors, prints them to
  stderr, and returns `""` (it lives in the providers layer, which the constraints
  forbid touching with logging). Those specific stderr lines are therefore not
  captured by the log. What IS captured durably, with full tracebacks: any
  exception that aborts a source/stage - including `NousEmbedder` timeouts (the
  embedder does NOT swallow, so its timeouts raise and are logged) and all DB
  faults. This is the honest, in-scope preservation; capturing the LLM's swallowed
  stderr would require either a stderr tee or changing the provider layer, both
  out of the stated constraints. Noted for a possible follow-up.
- **Editable install points at the main checkout**, not the worktree, so the suite
  must be run with `PYTHONPATH=<worktree>/src` for the changes to be exercised.
  Verified both the new and full suites this way.
- `curriculum doctor` blocks on the real Postgres connection probe when no DB is up
  (pre-existing behaviour, mocked in tests) - unrelated to this change; the new
  `log dir` probe itself is instant and side-effect free.
- Double-logging on a stage failure (stage-level `logger.exception` + terse
  CLI-level `logger.error`) is intentional and harmless: the traceback appears
  once, the CLI adds a one-line summary.

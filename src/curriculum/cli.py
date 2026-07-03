"""Agent-facing command line: the single entrypoint that makes the whole engine
reproducible and bootstrappable from a fresh checkout.

This is the outermost driver of the hexagon's *build* side (the MCP server is the
outermost driver of its *serve* side). It owns no business logic: every command
is a thin shell over :mod:`curriculum.app.build`, the docker-compose database, or
the MCP server. Its job is purely to turn argv + the environment-driven
:class:`curriculum.config.Settings` into one of those calls and a process exit
code, so a human or an agent can stand the system up with ``curriculum build
corpus.json`` instead of a pile of ad-hoc scripts.

Why every heavy import is deferred into the handlers
----------------------------------------------------
``curriculum --help`` and ``curriculum doctor`` MUST work on a machine that has
neither ``psycopg`` nor the ``mcp`` SDK installed -- that is the first thing an
operator runs to find out *what is missing*. So this module imports only the
standard library plus the light, dependency-free ``config``/``errors`` modules at
load time; :mod:`curriculum.app.build` (which can pull in the Postgres adapter)
and the MCP server are imported INSIDE the command handlers that need them. A
fresh-checkout ``doctor`` therefore diagnoses the very dependencies the other
commands require, rather than crashing on an ImportError before it can report.

Exit codes are returned, never raised: ``main`` converts argparse's ``SystemExit``
(``--help`` -> 0, a usage error -> 2) and any adapter/driver failure into an
integer, so callers, tests, and shell scripts get a clean status without having
to catch exceptions.

Standard library only.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from .config import Settings
from .config import load as load_settings
from .domain.errors import CurriculumError

__all__ = ["main"]


# --------------------------------------------------------------------------- #
# Small shared helpers.
# --------------------------------------------------------------------------- #
def _emit(obj: object) -> None:
    """Print a result as stable, indented JSON.

    ``sort_keys`` makes the output deterministic so an agent (or a test) can diff
    successive runs, and JSON keeps the machine-readable contract uniform across
    every data-returning command.
    """
    print(json.dumps(obj, indent=2, sort_keys=True))


def _course(args: argparse.Namespace, settings: Settings) -> str:
    """Resolve the course: an explicit ``--course`` wins, else the configured
    default, so single-course setups never have to pass the flag."""
    return args.course or settings.default_course


def _repo_root() -> Path | None:
    """Locate the directory that owns ``docker-compose.yml`` (the repo root).

    The compose file is a repo artifact, not something installed alongside the
    package, so we search upward from the working directory first (the operator
    usually runs ``curriculum db-up`` from the checkout) and then from this
    module's own location (covers an editable install invoked from elsewhere).
    ``None`` means no compose file was found -- the db commands then fail with an
    actionable message instead of running compose in the wrong place.
    """
    for start in (Path.cwd(), Path(__file__).resolve().parent):
        for directory in (start, *start.parents):
            if (directory / "docker-compose.yml").is_file():
                return directory
    return None


def _compose(compose_args: list[str]) -> int:
    """Run ``docker compose <compose_args>`` in the repo root, returning its code.

    Guards both preconditions up front -- the docker binary and the compose file
    -- because a missing-binary ``FileNotFoundError`` or a wrong-directory compose
    run are exactly the confusing failures this wrapper exists to prevent.
    """
    if shutil.which("docker") is None:
        print("curriculum: docker not found on PATH", file=sys.stderr)
        return 1
    root = _repo_root()
    if root is None:
        print(
            "curriculum: docker-compose.yml not found "
            "(run from the repository checkout)",
            file=sys.stderr,
        )
        return 1
    result = subprocess.run(["docker", "compose", *compose_args], cwd=str(root))
    return result.returncode


# --------------------------------------------------------------------------- #
# Build-side command handlers (each lazy-imports curriculum.app.build).
# --------------------------------------------------------------------------- #
def _cmd_db_up(args: argparse.Namespace, settings: Settings) -> int:
    """Start the bundled Postgres+pgvector service in the background."""
    return _compose(["up", "-d", "db"])


def _cmd_db_down(args: argparse.Namespace, settings: Settings) -> int:
    """Stop the compose stack (leaves the named data volume intact)."""
    return _compose(["down"])


def _cmd_status(args: argparse.Namespace, settings: Settings) -> int:
    """Print the read-only graph counts for a course (no inference, no writes)."""
    from .app import build

    _emit(build.status(settings, _course(args, settings)))
    return 0


def _cmd_ingest(args: argparse.Namespace, settings: Settings) -> int:
    """Ingest a manifest's sources into the concept/edge graph."""
    from .app import build

    manifest = build.load_manifest(args.manifest)
    _emit(build.ingest(manifest, settings))
    return 0


def _cmd_link(args: argparse.Namespace, settings: Settings) -> int:
    """Link isolated concepts via embedding-guided edge repair."""
    from .app import build

    _emit(build.link(settings, _course(args, settings)))
    return 0


def _cmd_questions(args: argparse.Namespace, settings: Settings) -> int:
    """Generate exam questions over the persisted graph (batched)."""
    from .app import build

    _emit(build.generate_questions(settings, _course(args, settings)))
    return 0


def _cmd_build(args: argparse.Namespace, settings: Settings) -> int:
    """Run the full pipeline for a manifest: ingest -> link -> questions.

    The course for the link/question stages comes from the manifest itself (not
    ``--course``): a build is scoped to exactly the course it ingests. Each
    stage's result is emitted as it completes so a long build shows progress
    rather than going dark until the end.
    """
    from .app import build

    manifest = build.load_manifest(args.manifest)
    course = manifest["course"]
    _emit({"stage": "ingest", "result": build.ingest(manifest, settings)})
    _emit({"stage": "link", "result": build.link(settings, course)})
    _emit(
        {
            "stage": "questions",
            "result": build.generate_questions(settings, course),
        }
    )
    return 0


# --------------------------------------------------------------------------- #
# Motivation-layer command handlers (each lazy-imports the composition root).
# --------------------------------------------------------------------------- #
def _render_check(payload: Mapping[str, Any]) -> str:
    """Render a ``checkin`` payload as the 30-second, gain-framed check-in.

    Pure and deterministic: no I/O, no service call -- everything it needs is
    already in ``payload``. Every line reports something the learner HAS
    (stability held, concepts unlocked, reviews ripe for today) rather than
    something owed, so this must never read like a list of obligations.
    """
    course = payload["course"]
    stability = payload["stability_days"]
    delta = payload.get("delta_since_last_check")
    consolidation = payload["consolidation"]
    ripe = payload["ripeness"]
    unlocks_ready = payload["unlocks_ready"]
    near_unlocks = payload["near_unlocks"]

    lines = [f"Course: {course}"]
    if delta is None:
        lines.append(f"Knowledge held: {stability:.1f} stability-days (first check)")
    else:
        sign = "+" if delta >= 0 else ""
        lines.append(
            f"Knowledge held: {stability:.1f} stability-days "
            f"({sign}{delta:.1f} since last check)"
        )
    lines.append(
        f"Holding steady: {consolidation['holding']} concept(s), "
        f"{consolidation['reviewed_since']} reviewed since last check"
    )
    lines.append(
        f"Ready today: {len(ripe['ready_now'])} "
        f"(tomorrow: {len(ripe['ready_tomorrow'])}, "
        f"this week: {len(ripe['ready_this_week'])})"
    )
    lines.append(f"Unlocked: {len(unlocks_ready)} new concept(s) available to start")
    one_away = sum(1 for row in near_unlocks if row["one_away"])
    if one_away:
        lines.append(f"On the verge: {one_away} concept(s) one step from unlocking")
    return "\n".join(lines)


def _cmd_check(args: argparse.Namespace, settings: Settings) -> int:
    """The 30-second deterministic check-in: no LLM, instant, gain-framed.

    ``--json`` bypasses the human render entirely and emits the raw payload,
    so an agent can consume the same numbers a human reads on screen.
    """
    from .application.composition import build_service

    service = build_service(settings)
    payload = service.checkin(_course(args, settings))
    if args.json:
        _emit(payload)
    else:
        print(_render_check(payload))
    return 0


def _cmd_flag_question(args: argparse.Namespace, settings: Settings) -> int:
    """Retire a question so it is never served again (the kill switch)."""
    from .application.composition import build_service

    service = build_service(settings)
    _emit(service.flag_question(args.question_id, reason=args.reason))
    return 0


# --------------------------------------------------------------------------- #
# Serve-side and registration handlers.
# --------------------------------------------------------------------------- #
def _cmd_serve(args: argparse.Namespace, settings: Settings) -> int:
    """Become the stdio MCP server (this is what Hermes launches).

    ``os.execv`` REPLACES this process with ``python -m curriculum.mcp.server``,
    so Hermes talks to the server directly over the stdio it opened for
    ``curriculum serve`` -- no wrapper process sits between them holding the pipes.
    Nothing is printed (stdout is the MCP transport). On success execv does not
    return; the trailing ``return`` only covers the impossible fall-through, and
    an ``OSError`` propagates to ``main`` to become a non-zero exit.
    """
    os.execv(sys.executable, [sys.executable, "-m", "curriculum.mcp.server"])
    return 0  # pragma: no cover - unreachable: execv never returns on success


def _register_argv(python: str, settings: Settings, key_value: str) -> list[str]:
    """Build the ``hermes mcp add`` argv that registers this MCP server.

    The bundle path is absolutised because Hermes spawns the server from its own
    (arbitrary) working directory, so a relative ``./bundle`` would not resolve.
    ``key_value`` is injected by the caller so the runnable form can carry the
    real key while the printed form carries a shell reference (see
    :func:`_cmd_mcp_register`).
    """
    bundle = os.path.abspath(settings.okf_bundle_path)
    return [
        "hermes", "mcp", "add", "curriculum",
        "--command", python,
        "--env",
        f"NOUS_API_KEY={key_value}",
        f"CURRICULUM_DB_URL={settings.database_url}",
        f"CURRICULUM_OKF_PATH={bundle}",
        "--args", "-m", "curriculum.mcp.server",
    ]


def _render(argv: list[str]) -> str:
    """Render an argv as a copy-pasteable shell line, keeping the key a reference.

    Every token is shell-quoted so paths with spaces survive, EXCEPT the
    ``NOUS_API_KEY`` assignment, which is emitted as ``NOUS_API_KEY="$NOUS_API_KEY"``
    so the secret never lands in printed output (the user's shell expands it at
    paste time) while the line stays runnable.
    """
    parts: list[str] = []
    for token in argv:
        if token.startswith("NOUS_API_KEY="):
            parts.append('NOUS_API_KEY="$NOUS_API_KEY"')
        else:
            parts.append(shlex.quote(token))
    return " ".join(parts)


def _cmd_mcp_register(args: argparse.Namespace, settings: Settings) -> int:
    """Register this MCP server with Hermes, or print the command if hermes is absent.

    Convenience wiring for the one supported host (Hermes). When ``hermes`` is on
    PATH we run the registration with the resolved key so it actually lands; when
    it is not, we print the exact command (key shown as a shell reference) for the
    operator to run wherever Hermes lives. The printed form is shown in both cases
    so the operator can see what ran without the secret being echoed.
    """
    python = sys.executable
    printable = _render(_register_argv(python, settings, ""))
    if shutil.which("hermes") is None:
        print("hermes not found on PATH; run this where Hermes is installed:")
        print(printable)
        return 0
    print(f"running: {printable}")
    result = subprocess.run(
        _register_argv(python, settings, settings.nous_api_key or "")
    )
    return result.returncode


# --------------------------------------------------------------------------- #
# doctor: a fresh-machine readiness checklist (each probe is isolated + safe).
# --------------------------------------------------------------------------- #
def _check_docker() -> tuple[str, bool, str]:
    """Is the docker binary available (needed for ``db-up``)?"""
    path = shutil.which("docker")
    if path:
        return ("docker", True, path)
    return ("docker", False, "not found on PATH")


def _check_db(settings: Settings) -> tuple[str, bool, str]:
    """Can we open the configured database and run a trivial query?

    The Postgres adapter is imported lazily and EVERY failure mode is folded into
    a reported line -- driver missing (``connect`` raises a RuntimeError), refused
    connection, or a failing probe query -- so ``doctor`` never raises; it only
    reports. The connection is always closed.
    """
    try:
        from .storage.postgres import connect
    except ImportError as exc:  # pragma: no cover - module imports without driver
        return ("database", False, f"adapter unavailable: {exc}")
    try:
        connection = connect(settings.database_url)
    except Exception as exc:  # noqa: BLE001 - driver-missing or refused: report, not raise
        return ("database", False, f"unreachable: {exc}")
    try:
        connection.execute("SELECT 1")
    except Exception as exc:  # noqa: BLE001 - connected but unhealthy
        return ("database", False, f"query failed: {exc}")
    finally:
        try:
            connection.close()
        except Exception:  # noqa: BLE001 - close errors are not actionable here
            pass
    return ("database", True, settings.database_url)


def _check_nous(settings: Settings) -> tuple[str, bool, str]:
    """Is the Nous API key set (required by every inference-backed build stage)?"""
    if settings.nous_api_key:
        return ("NOUS_API_KEY", True, "set")
    return ("NOUS_API_KEY", False, "not set (export NOUS_API_KEY)")


def _check_bundle(settings: Settings) -> tuple[str, bool, str]:
    """Does the OKF content bundle directory exist yet?

    Reported as missing before the first build (the ingest step creates it), so a
    pre-build ``doctor`` honestly shows it is not there rather than implying the
    corpus is ready.
    """
    path = os.path.abspath(settings.okf_bundle_path)
    if os.path.isdir(path):
        return ("okf bundle", True, path)
    return ("okf bundle", False, f"{path} (missing; created on first ingest)")


def _cmd_doctor(args: argparse.Namespace, settings: Settings) -> int:
    """Print an OK/MISS checklist of the build prerequisites; non-zero if any miss.

    Returns a non-zero code when something is missing so the command doubles as a
    scriptable readiness gate, while still printing the full picture either way.
    """
    checks = [
        _check_docker(),
        _check_db(settings),
        _check_nous(settings),
        _check_bundle(settings),
    ]
    all_ok = True
    for label, ok, detail in checks:
        mark = " OK " if ok else "MISS"
        print(f"[{mark}] {label}: {detail}")
        all_ok = all_ok and ok
    return 0 if all_ok else 1


# --------------------------------------------------------------------------- #
# Parser wiring + entrypoint.
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    """Construct the argparse parser and bind each subcommand to its handler.

    Subcommands carry their handler via ``set_defaults(func=...)`` so ``main``
    dispatches uniformly; the top parser defaults ``func`` to ``None`` so a
    no-subcommand invocation is detectable (and answered with the help text)
    rather than crashing on a missing attribute.
    """
    parser = argparse.ArgumentParser(
        prog="curriculum",
        description=(
            "Build and serve the curriculum knowledge-graph engine: stand up the "
            "database, ingest a corpus manifest into the graph, link and generate "
            "questions, and serve the result to Hermes over MCP."
        ),
    )
    parser.set_defaults(func=None)
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    # A shared --course option for the course-scoped read/build steps.
    course_parent = argparse.ArgumentParser(add_help=False)
    course_parent.add_argument(
        "--course",
        default=None,
        help="course name (defaults to settings.default_course)",
    )

    p_db_up = sub.add_parser("db-up", help="start the Postgres+pgvector container")
    p_db_up.set_defaults(func=_cmd_db_up)

    p_db_down = sub.add_parser("db-down", help="stop the compose stack")
    p_db_down.set_defaults(func=_cmd_db_down)

    p_status = sub.add_parser(
        "status", parents=[course_parent], help="print graph counts for a course"
    )
    p_status.set_defaults(func=_cmd_status)

    p_ingest = sub.add_parser("ingest", help="ingest a manifest into the graph")
    p_ingest.add_argument("manifest", help="path to the corpus manifest JSON")
    p_ingest.set_defaults(func=_cmd_ingest)

    p_link = sub.add_parser(
        "link", parents=[course_parent], help="link isolated concepts"
    )
    p_link.set_defaults(func=_cmd_link)

    p_questions = sub.add_parser(
        "questions", parents=[course_parent], help="generate exam questions"
    )
    p_questions.set_defaults(func=_cmd_questions)

    p_build = sub.add_parser(
        "build", help="full pipeline: ingest -> link -> questions"
    )
    p_build.add_argument("manifest", help="path to the corpus manifest JSON")
    p_build.set_defaults(func=_cmd_build)

    p_check = sub.add_parser(
        "check",
        parents=[course_parent],
        help="30-second gain-framed check-in (no LLM)",
    )
    p_check.add_argument(
        "--json", action="store_true", help="emit the raw payload as JSON"
    )
    p_check.set_defaults(func=_cmd_check)

    p_flag_question = sub.add_parser(
        "flag-question", help="retire a question so it is never served again"
    )
    p_flag_question.add_argument("question_id", help="id of the question to retire")
    p_flag_question.add_argument(
        "--reason", default="", help="optional reason for the flag"
    )
    p_flag_question.set_defaults(func=_cmd_flag_question)

    p_serve = sub.add_parser("serve", help="run the stdio MCP server (for Hermes)")
    p_serve.set_defaults(func=_cmd_serve)

    p_register = sub.add_parser(
        "mcp-register", help="register this MCP server with Hermes"
    )
    p_register.set_defaults(func=_cmd_mcp_register)

    p_doctor = sub.add_parser(
        "doctor", help="check prerequisites (docker, DB, key, bundle)"
    )
    p_doctor.set_defaults(func=_cmd_doctor)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse ``argv`` and run the chosen command, returning a process exit code.

    Argparse's ``SystemExit`` is caught and converted to its integer code so that
    ``--help`` (0) and usage errors (2) return cleanly instead of raising. With no
    subcommand we print the help text and return 0. All command/adapter failures
    are funnelled into a single non-zero exit with a concise stderr message:
    :class:`CurriculumError` for domain problems (bad manifest, missing key) and a
    broad catch for driver/DB/subprocess failures, since the CLI is a boundary and
    a clean status beats a stack trace.
    """
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code) if exc.code is not None else 0

    if getattr(args, "func", None) is None:
        parser.print_help()
        return 0

    try:
        settings = load_settings()
        return args.func(args, settings)
    except CurriculumError as exc:
        print(f"curriculum: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - boundary: driver/DB/subprocess -> clean exit
        print(f"curriculum: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover - process entrypoint
    sys.exit(main())

"""Runtime settings, loaded from the environment.

Single source of configuration; adapters receive their settings explicitly
(no global lookups deep in the code). Defaults assume the bundled
docker-compose Postgres on port 5433.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Settings:
    database_url: str = "postgresql://curriculum:curriculum@localhost:5433/curriculum"
    okf_bundle_path: str = "./bundle"
    default_course: str = "Cybersecurity"
    # Provider-agnostic inference credentials. Any OpenAI-compatible endpoint
    # (Nous, NVIDIA NIM, vLLM, ...) is addressed by ``api_key`` + ``base_url``;
    # the vendor default keeps existing Nous deployments working out of the box.
    api_key: str | None = None
    base_url: str = "https://inference-api.nousresearch.com/v1"
    ingest_model: str = "deepseek/deepseek-v4-flash"
    embed_model: str = "google/gemini-embedding-2"  # multimodal, native 3072-dim
    embedding_dim: int = 3072
    # Completion budget per LLM call. Reasoning models spend this SAME budget on
    # hidden thinking before any visible content, so a value that is ample for a
    # plain model can starve a reasoning one into empty completions -- raise this
    # (e.g. 32768) when ingesting with a reasoning model.
    max_tokens: int = 8192
    # Where per-invocation build logs land (see curriculum.app.build_logging).
    # Relative to the working directory by default so an operator finds them next
    # to where they launched the build.
    log_dir: str = "logs"


def load(env: dict[str, str] | None = None) -> Settings:
    e = os.environ if env is None else env
    d = Settings()  # instance access yields real defaults (class access yields slot descriptors)
    return Settings(
        database_url=e.get("CURRICULUM_DB_URL", d.database_url),
        okf_bundle_path=e.get("CURRICULUM_OKF_PATH", d.okf_bundle_path),
        default_course=e.get("CURRICULUM_COURSE", d.default_course),
        # Generic names are primary; the legacy NOUS_* names remain as
        # backward-compatible fallbacks. The generic name wins when both are set
        # (``or`` also treats an empty generic value as unset, deferring to the
        # legacy name and finally the default).
        api_key=e.get("CURRICULUM_API_KEY") or e.get("NOUS_API_KEY") or d.api_key,
        base_url=e.get("CURRICULUM_BASE_URL") or e.get("NOUS_BASE_URL") or d.base_url,
        ingest_model=e.get("CURRICULUM_INGEST_MODEL", d.ingest_model),
        embed_model=e.get("CURRICULUM_EMBED_MODEL", d.embed_model),
        embedding_dim=int(e.get("CURRICULUM_EMBED_DIM", str(d.embedding_dim))),
        max_tokens=int(e.get("CURRICULUM_MAX_TOKENS", str(d.max_tokens))),
        log_dir=e.get("CURRICULUM_LOG_DIR", d.log_dir),
    )

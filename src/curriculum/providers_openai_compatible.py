"""OpenAI-compatible LlmProvider/EmbeddingProvider -- the only place paid
inference happens.

Speaks the OpenAI-compatible HTTP surface (``/chat/completions`` and
``/embeddings``) over the standard library ``urllib`` (no extra dependency), so
it works against any endpoint that implements that surface -- Nous, NVIDIA NIM,
vLLM, and similar. The endpoint is chosen purely by ``base_url`` + ``api_key``;
nothing here is vendor-specific. Tests and dry runs use
:class:`curriculum.providers_fake.FakeLlm` instead.

Resilience: a single failed call returns "" rather than raising, so one bad
chunk does not abort a whole ingest (the pass then sees "no items"). Failures
are reported two ways with the SAME text: a stderr print for interactive runs,
and a WARNING on the module-level ``curriculum.providers`` logger so the durable
build log (issue #3) captures them -- otherwise a build where every call times
out would swallow every failure and log each source as healthy while silently
producing nothing. The logger does NOT propagate to the root handler, so the two
channels never double up on the console.
"""
from __future__ import annotations

import json
import logging
import sys
import urllib.error
import urllib.request
from typing import Sequence

from .ports.providers import EmbeddingProvider, LlmProvider

# Shared, module-level logger. The build-logging layer attaches its per-invocation
# FileHandler here so swallowed provider failures reach the durable log. propagate
# is False so a WARNING never reaches the root's last-resort stderr handler and
# thus never duplicates the interactive print below on the console.
_LOGGER = logging.getLogger("curriculum.providers")
_LOGGER.propagate = False
# A NullHandler keeps ``callHandlers`` from falling back to logging's last-resort
# stderr handler when no build log is attached -- that fallback would print the
# WARNING a second time next to the interactive stderr line below. The build log
# attaches its own FileHandler alongside this no-op one.
_LOGGER.addHandler(logging.NullHandler())


def _report_failure(message: str) -> None:
    """Surface a swallowed provider failure on BOTH channels with one text.

    stderr keeps interactive runs visible; the ``curriculum.providers`` WARNING is
    what the durable build log records. The message is passed as a ``%s`` argument
    so any ``%`` in an upstream body excerpt is never treated as a format spec.
    """
    print(message, file=sys.stderr)
    _LOGGER.warning("%s", message)


class OpenAICompatibleLlm(LlmProvider):
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://inference-api.nousresearch.com/v1",
        model: str = "deepseek/deepseek-v4-flash",
        timeout: float = 120.0,
    ) -> None:
        self._key = api_key
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._model = model
        self._timeout = timeout
        self.calls = 0
        self.failures = 0

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 8192,
        temperature: float = 0.2,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        body = json.dumps(
            {
                "model": self._model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            self._url,
            data=body,
            headers={
                "Authorization": f"Bearer {self._key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        self.calls += 1
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            self.failures += 1
            detail = exc.read().decode("utf-8", "ignore")[:200]
            _report_failure(f"[OpenAICompatibleLlm] HTTP {exc.code}: {detail}")
            return ""
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            self.failures += 1
            _report_failure(f"[OpenAICompatibleLlm] request failed: {exc}")
            return ""
        try:
            return payload["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, TypeError):  # pragma: no cover
            self.failures += 1
            _report_failure(
                f"[OpenAICompatibleLlm] unexpected response shape: {str(payload)[:200]}"
            )
            return ""


class OpenAICompatibleEmbedder(EmbeddingProvider):
    """OpenAI-compatible embeddings (``/embeddings``), batched.

    Default model ``google/gemini-embedding-2`` returns 3072-dim vectors,
    matching the schema's ``vector(3072)``; it is a top-tier (and multimodal)
    embedder, so the same vectors can later cover lesson images/diagrams, not
    just text. Embeddings are cheap (fractions of a cent for a whole course), so
    they are computed online rather than via a shipped local model. ``embed``
    preserves input order using the per-row ``index`` the API returns.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://inference-api.nousresearch.com/v1",
        model: str = "google/gemini-embedding-2",
        dim: int = 3072,
        timeout: float = 60.0,
        batch: int = 64,
    ) -> None:
        self._key = api_key
        self._url = base_url.rstrip("/") + "/embeddings"
        self._model = model
        self.dim = dim
        self._timeout = timeout
        self._batch = max(1, batch)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        # Empty text confuses some embedding models; substitute a single space.
        items = [t if t and t.strip() else " " for t in texts]
        out: list[list[float]] = []
        for i in range(0, len(items), self._batch):
            out.extend(self._embed_batch(items[i : i + self._batch]))
        return out

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        body = json.dumps({"model": self._model, "input": batch}).encode("utf-8")
        req = urllib.request.Request(
            self._url,
            data=body,
            headers={
                "Authorization": f"Bearer {self._key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        rows = sorted(payload["data"], key=lambda d: d.get("index", 0))
        return [list(row["embedding"]) for row in rows]

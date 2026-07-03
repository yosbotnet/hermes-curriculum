"""Tests for the OpenAI-compatible HTTP providers.

Stdlib unittest only and strictly offline: the HTTP layer (``urllib.request``)
is faked so no socket is ever opened. What these pin is the load-bearing
*request construction* -- the URL, method, auth header, and JSON body the
providers send -- because that shape is the actual contract with any
OpenAI-compatible endpoint (Nous, NVIDIA NIM, vLLM, ...). Response parsing and
the order-preserving embedding merge are covered too.

The legacy-import test proves that ``curriculum.providers_nous`` keeps exposing
the same public names after the implementation moved modules, so existing
imports do not break.
"""
from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stderr
from unittest import mock

from curriculum.providers_openai_compatible import (
    OpenAICompatibleEmbedder,
    OpenAICompatibleLlm,
)


class _FakeResponse:
    """Minimal stand-in for the object ``urllib.request.urlopen`` yields."""

    def __init__(self, payload: dict) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc) -> None:
        return None


def _patch_urlopen(payload: dict):
    """Patch urlopen to capture the Request and return ``payload``."""
    captured: dict = {}

    def fake_urlopen(req, timeout=None):
        captured["request"] = req
        captured["timeout"] = timeout
        return _FakeResponse(payload)

    patcher = mock.patch(
        "curriculum.providers_openai_compatible.urllib.request.urlopen",
        side_effect=fake_urlopen,
    )
    return patcher, captured


class LlmRequestShapeTest(unittest.TestCase):
    def test_chat_completion_request_shape(self) -> None:
        payload = {"choices": [{"message": {"content": "hello"}}]}
        patcher, captured = _patch_urlopen(payload)
        llm = OpenAICompatibleLlm(
            api_key="secret-key",
            base_url="https://integrate.api.nvidia.com/v1",
            model="vendor/model-a",
        )
        with patcher:
            out = llm.complete(
                "the prompt", system="be terse", max_tokens=256, temperature=0.5
            )

        self.assertEqual(out, "hello")
        req = captured["request"]
        # Endpoint: base_url + /chat/completions, POST + bearer auth + JSON.
        self.assertEqual(
            req.full_url, "https://integrate.api.nvidia.com/v1/chat/completions"
        )
        self.assertEqual(req.get_method(), "POST")
        self.assertEqual(req.headers["Authorization"], "Bearer secret-key")
        self.assertEqual(req.headers["Content-type"], "application/json")

        body = json.loads(req.data.decode("utf-8"))
        self.assertEqual(body["model"], "vendor/model-a")
        self.assertEqual(body["temperature"], 0.5)
        self.assertEqual(body["max_tokens"], 256)
        self.assertEqual(
            body["messages"],
            [
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "the prompt"},
            ],
        )

    def test_system_message_omitted_when_absent(self) -> None:
        payload = {"choices": [{"message": {"content": "x"}}]}
        patcher, captured = _patch_urlopen(payload)
        llm = OpenAICompatibleLlm(api_key="k", model="m")
        with patcher:
            llm.complete("only user")
        body = json.loads(captured["request"].data.decode("utf-8"))
        self.assertEqual(body["messages"], [{"role": "user", "content": "only user"}])

    def test_base_url_trailing_slash_is_normalised(self) -> None:
        patcher, captured = _patch_urlopen(
            {"choices": [{"message": {"content": ""}}]}
        )
        llm = OpenAICompatibleLlm(api_key="k", base_url="https://host/v1/", model="m")
        with patcher:
            llm.complete("p")
        self.assertEqual(
            captured["request"].full_url, "https://host/v1/chat/completions"
        )


class EmbedderRequestShapeTest(unittest.TestCase):
    def test_embeddings_request_shape_and_order(self) -> None:
        # Return rows out of order to prove the provider re-sorts by index.
        payload = {
            "data": [
                {"index": 1, "embedding": [0.4, 0.5]},
                {"index": 0, "embedding": [0.1, 0.2]},
            ]
        }
        patcher, captured = _patch_urlopen(payload)
        embedder = OpenAICompatibleEmbedder(
            api_key="secret-key",
            base_url="https://integrate.api.nvidia.com/v1",
            model="vendor/embed-b",
            dim=2,
        )
        with patcher:
            out = embedder.embed(["alpha", "beta"])

        req = captured["request"]
        self.assertEqual(
            req.full_url, "https://integrate.api.nvidia.com/v1/embeddings"
        )
        self.assertEqual(req.get_method(), "POST")
        self.assertEqual(req.headers["Authorization"], "Bearer secret-key")
        body = json.loads(req.data.decode("utf-8"))
        self.assertEqual(body["model"], "vendor/embed-b")
        self.assertEqual(body["input"], ["alpha", "beta"])
        # Order preserved via the per-row index.
        self.assertEqual(out, [[0.1, 0.2], [0.4, 0.5]])

    def test_blank_text_substituted_with_space(self) -> None:
        payload = {"data": [{"index": 0, "embedding": [1.0]}]}
        patcher, captured = _patch_urlopen(payload)
        embedder = OpenAICompatibleEmbedder(api_key="k", model="m", dim=1)
        with patcher:
            embedder.embed([""])
        body = json.loads(captured["request"].data.decode("utf-8"))
        self.assertEqual(body["input"], [" "])


class LlmFailureLoggingTest(unittest.TestCase):
    """Issue #3 follow-up: a swallowed LLM failure must still be visible.

    The provider returns "" on a request failure (so one bad chunk does not abort
    the whole ingest), but that failure MUST also reach the durable build log.
    It does so through a module-level ``curriculum.providers`` WARNING record with
    the same diagnostic the interactive stderr line carries.
    """

    def _failing_llm(self, boom):
        patcher = mock.patch(
            "curriculum.providers_openai_compatible.urllib.request.urlopen",
            side_effect=boom,
        )
        return OpenAICompatibleLlm(api_key="k", model="m"), patcher

    def test_request_failure_emits_provider_warning_and_returns_empty(self) -> None:
        def boom(req, timeout=None):
            raise TimeoutError("timed out")

        llm, patcher = self._failing_llm(boom)
        # redirect_stderr keeps the interactive print out of the test report;
        # it does not affect the logger assertion.
        with patcher, redirect_stderr(io.StringIO()):
            with self.assertLogs("curriculum.providers", level="WARNING") as captured:
                out = llm.complete("p")

        self.assertEqual(out, "")  # still swallowed
        self.assertEqual(llm.failures, 1)
        self.assertTrue(
            any("request failed" in line for line in captured.output),
            captured.output,
        )

    def test_http_error_emits_provider_warning(self) -> None:
        import urllib.error

        def boom(req, timeout=None):
            raise urllib.error.HTTPError(
                url="http://x", code=503, msg="down", hdrs=None,
                fp=io.BytesIO(b"upstream unavailable"),
            )

        llm, patcher = self._failing_llm(boom)
        with patcher, redirect_stderr(io.StringIO()):
            with self.assertLogs("curriculum.providers", level="WARNING") as captured:
                out = llm.complete("p")

        self.assertEqual(out, "")
        self.assertTrue(
            any("HTTP 503" in line for line in captured.output), captured.output
        )

    def test_interactive_stderr_message_is_not_duplicated_on_console(self) -> None:
        # Console visibility must not regress: the failure prints exactly ONCE to
        # stderr (the provider logger does not propagate to the root handler).
        def boom(req, timeout=None):
            raise TimeoutError("timed out")

        llm, patcher = self._failing_llm(boom)
        stderr = io.StringIO()
        with patcher, redirect_stderr(stderr):
            llm.complete("p")
        self.assertEqual(stderr.getvalue().count("request failed"), 1)


class LegacyShimImportTest(unittest.TestCase):
    """Acceptance criterion 4: legacy import compatibility."""

    def test_providers_nous_exposes_the_same_classes(self) -> None:
        from curriculum import providers_nous

        self.assertIs(providers_nous.NousLlm, OpenAICompatibleLlm)
        self.assertIs(providers_nous.NousEmbedder, OpenAICompatibleEmbedder)

    def test_legacy_names_are_still_constructible(self) -> None:
        from curriculum.providers_nous import NousEmbedder, NousLlm

        self.assertIsInstance(NousLlm(api_key="k"), OpenAICompatibleLlm)
        self.assertIsInstance(NousEmbedder(api_key="k"), OpenAICompatibleEmbedder)


if __name__ == "__main__":
    unittest.main()


class EmptyCompletionDiagnosisTest(unittest.TestCase):
    """HTTP 200 with no visible content must be diagnosed, not swallowed.

    Reasoning models spend max_tokens on hidden thinking; an exhausted budget
    returns finish_reason "length" and an EMPTY content field. Before this
    diagnosis existed, that flowed through the lenient parsing as "no items"
    and a fully starved build logged every source as healthy.
    """

    def _complete(self, payload: dict) -> tuple[str, OpenAICompatibleLlm, str]:
        patcher, _ = _patch_urlopen(payload)
        llm = OpenAICompatibleLlm(api_key="k")
        stderr = io.StringIO()
        with patcher, redirect_stderr(stderr), self.assertLogs(
            "curriculum.providers", level="WARNING"
        ) as logs:
            out = llm.complete("prompt")
        self.assertEqual(logs.output and len(logs.output), 1)
        return out, llm, logs.output[0]

    def test_reasoning_budget_exhaustion_is_diagnosed(self) -> None:
        payload = {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {"content": "", "reasoning_content": "thinking..."},
                }
            ]
        }
        out, llm, warning = self._complete(payload)
        self.assertEqual(out, "")
        self.assertEqual(llm.failures, 1)
        self.assertIn("EMPTY completion", warning)
        self.assertIn("CURRICULUM_MAX_TOKENS", warning)
        self.assertIn("length", warning)

    def test_reasoning_content_without_length_is_still_diagnosed(self) -> None:
        payload = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": None, "reasoning": "hidden thoughts"},
                }
            ]
        }
        out, llm, warning = self._complete(payload)
        self.assertEqual(out, "")
        self.assertEqual(llm.failures, 1)
        self.assertIn("reasoning tokens present", warning)

    def test_plain_empty_completion_warns_generically(self) -> None:
        payload = {
            "choices": [{"finish_reason": "stop", "message": {"content": ""}}]
        }
        out, llm, warning = self._complete(payload)
        self.assertEqual(out, "")
        self.assertEqual(llm.failures, 1)
        self.assertIn("empty completion", warning)

    def test_whitespace_only_content_counts_as_empty(self) -> None:
        payload = {
            "choices": [{"finish_reason": "stop", "message": {"content": "  \n"}}]
        }
        out, llm, _warning = self._complete(payload)
        self.assertEqual(out, "")
        self.assertEqual(llm.failures, 1)

    def test_constructor_max_tokens_is_sent_when_call_omits_it(self) -> None:
        payload = {"choices": [{"message": {"content": "fine"}}]}
        patcher, captured = _patch_urlopen(payload)
        llm = OpenAICompatibleLlm(api_key="k", max_tokens=32768)
        with patcher:
            out = llm.complete("prompt")
        self.assertEqual(out, "fine")
        body = json.loads(captured["request"].data.decode("utf-8"))
        self.assertEqual(body["max_tokens"], 32768)

    def test_call_site_max_tokens_still_overrides_constructor(self) -> None:
        payload = {"choices": [{"message": {"content": "fine"}}]}
        patcher, captured = _patch_urlopen(payload)
        llm = OpenAICompatibleLlm(api_key="k", max_tokens=32768)
        with patcher:
            llm.complete("prompt", max_tokens=64)
        body = json.loads(captured["request"].data.decode("utf-8"))
        self.assertEqual(body["max_tokens"], 64)

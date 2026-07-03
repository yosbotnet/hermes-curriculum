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

import json
import unittest
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

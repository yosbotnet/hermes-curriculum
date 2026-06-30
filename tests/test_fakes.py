"""Tests for the deterministic fake providers.

Stdlib unittest only. The point of these fakes is determinism and zero cost, so
the assertions hammer on: same input -> identical output, distinct inputs ->
distinct outputs, correct shape, unit-norm embeddings, and the script/default
contract of the fake LLM.
"""
from __future__ import annotations

import math
import unittest

from curriculum.ports.providers import EmbeddingProvider, LlmProvider
from curriculum.providers_fake import FakeEmbedder, FakeLlm


def _l2_norm(vector: list[float]) -> float:
    return math.sqrt(sum(component * component for component in vector))


class FakeEmbedderTest(unittest.TestCase):
    def test_is_embedding_provider(self) -> None:
        self.assertIsInstance(FakeEmbedder(), EmbeddingProvider)

    def test_default_and_custom_dim_are_set(self) -> None:
        self.assertEqual(FakeEmbedder().dim, 64)
        self.assertEqual(FakeEmbedder(dim=8).dim, 8)

    def test_invalid_dim_rejected(self) -> None:
        with self.assertRaises(ValueError):
            FakeEmbedder(dim=0)
        with self.assertRaises(ValueError):
            FakeEmbedder(dim=-3)

    def test_returns_one_vector_per_text_in_order(self) -> None:
        embedder = FakeEmbedder(dim=16)
        out = embedder.embed(["alpha", "beta", "gamma"])
        self.assertEqual(len(out), 3)
        # Order is preserved: re-embedding a single text matches its slot.
        self.assertEqual(out[1], embedder.embed(["beta"])[0])

    def test_empty_input_yields_empty_output(self) -> None:
        self.assertEqual(FakeEmbedder().embed([]), [])

    def test_vector_has_requested_dim(self) -> None:
        for dim in (1, 8, 64, 100):
            vec = FakeEmbedder(dim=dim).embed(["concept-x"])[0]
            self.assertEqual(len(vec), dim)

    def test_identical_text_identical_vector(self) -> None:
        embedder = FakeEmbedder(dim=32)
        first = embedder.embed(["graph theory"])[0]
        second = embedder.embed(["graph theory"])[0]
        self.assertEqual(first, second)

    def test_deterministic_across_instances(self) -> None:
        # Determinism must not depend on instance identity or call history.
        a = FakeEmbedder(dim=32).embed(["dijkstra"])[0]
        b = FakeEmbedder(dim=32).embed(["dijkstra"])[0]
        self.assertEqual(a, b)

    def test_different_text_different_vector(self) -> None:
        embedder = FakeEmbedder(dim=32)
        a = embedder.embed(["hash table"])[0]
        b = embedder.embed(["hash tables"])[0]  # one char different
        self.assertNotEqual(a, b)

    def test_distinct_inputs_have_distinct_vectors_in_batch(self) -> None:
        texts = [f"concept-{i}" for i in range(25)]
        vectors = FakeEmbedder(dim=48).embed(texts)
        unique = {tuple(round(c, 9) for c in vec) for vec in vectors}
        self.assertEqual(len(unique), len(texts))

    def test_vectors_are_unit_norm(self) -> None:
        embedder = FakeEmbedder(dim=64)
        for text in ["", "a", "the quick brown fox", "unicode-free ascii"]:
            with self.subTest(text=text):
                self.assertAlmostEqual(_l2_norm(embedder.embed([text])[0]), 1.0, places=9)

    def test_empty_string_still_unit_norm(self) -> None:
        # The empty string is a real edge: it must still produce a usable,
        # normalised vector (not all-zeros, not a crash).
        vec = FakeEmbedder(dim=4).embed([""])[0]
        self.assertEqual(len(vec), 4)
        self.assertAlmostEqual(_l2_norm(vec), 1.0, places=9)

    def test_components_within_signed_unit_range(self) -> None:
        vec = FakeEmbedder(dim=64).embed(["bounds check"])[0]
        # After L2-normalisation the squared components sum to 1, so each one
        # individually has magnitude <= 1: every component stays within [-1, 1].
        for component in vec:
            self.assertLessEqual(component, 1.0)
            self.assertGreaterEqual(component, -1.0)


class FakeLlmTest(unittest.TestCase):
    def test_is_llm_provider(self) -> None:
        self.assertIsInstance(FakeLlm(), LlmProvider)

    def test_scripted_substring_match(self) -> None:
        llm = FakeLlm({"FOO": "scripted-foo"})
        self.assertEqual(llm.complete("please consider FOO carefully"), "scripted-foo")

    def test_no_match_returns_deterministic_default(self) -> None:
        llm = FakeLlm({"FOO": "scripted-foo"})
        first = llm.complete("nothing relevant here")
        second = llm.complete("nothing relevant here")
        self.assertEqual(first, second)
        self.assertNotEqual(first, "scripted-foo")

    def test_default_used_when_no_scripts(self) -> None:
        llm = FakeLlm()
        out = llm.complete("a bare prompt")
        self.assertTrue(out.startswith("[fake-llm]"))

    def test_none_scripts_behaves_like_empty(self) -> None:
        self.assertEqual(FakeLlm(None).complete("x"), FakeLlm({}).complete("x"))

    def test_different_prompts_differ(self) -> None:
        llm = FakeLlm()
        self.assertNotEqual(llm.complete("prompt one"), llm.complete("prompt two"))

    def test_default_is_independent_of_generation_params(self) -> None:
        # Determinism: system/max_tokens/temperature must not perturb the stub.
        llm = FakeLlm()
        base = llm.complete("stable prompt")
        self.assertEqual(base, llm.complete("stable prompt", system="you are a tutor"))
        self.assertEqual(base, llm.complete("stable prompt", max_tokens=4096))
        self.assertEqual(base, llm.complete("stable prompt", temperature=1.9))

    def test_scripted_value_independent_of_generation_params(self) -> None:
        llm = FakeLlm({"KEY": "answer"})
        self.assertEqual(
            llm.complete("xx KEY xx", system="s", max_tokens=10, temperature=0.0),
            "answer",
        )

    def test_longest_match_wins_when_multiple_keys_match(self) -> None:
        # Both "ab" and "abc" are substrings; the most-specific (longest) wins,
        # regardless of dict insertion order.
        self.assertEqual(
            FakeLlm({"ab": "short", "abc": "long"}).complete("zzabczz"), "long"
        )
        self.assertEqual(
            FakeLlm({"abc": "long", "ab": "short"}).complete("zzabczz"), "long"
        )

    def test_match_is_deterministic_for_equal_length_keys(self) -> None:
        # Equal-length competing triggers must resolve the same way every time.
        scripts = {"aa": "first", "bb": "second"}
        out = FakeLlm(scripts).complete("xx aa bb xx")
        self.assertEqual(out, FakeLlm(scripts).complete("xx aa bb xx"))
        self.assertIn(out, {"first", "second"})

    def test_no_substring_match_falls_through_to_default(self) -> None:
        llm = FakeLlm({"absent": "never"})
        out = llm.complete("this prompt lacks the trigger")
        self.assertNotEqual(out, "never")
        self.assertTrue(out.startswith("[fake-llm]"))

    def test_constructor_copies_scripts(self) -> None:
        # Mutating the caller's dict after construction must not change behaviour.
        original = {"KEY": "v1"}
        llm = FakeLlm(original)
        original["KEY"] = "v2"
        original["NEW"] = "v3"
        self.assertEqual(llm.complete("has KEY"), "v1")
        self.assertNotEqual(llm.complete("has NEW"), "v3")


if __name__ == "__main__":
    unittest.main()

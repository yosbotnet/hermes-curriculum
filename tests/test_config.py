"""Tests for environment-driven settings (curriculum.config).

Stdlib unittest only and pure: ``load`` is fed an explicit ``env`` dict, so no
process environment is read or mutated. These pin the provider-agnostic
configuration contract:

* the generic ``CURRICULUM_API_KEY`` / ``CURRICULUM_BASE_URL`` are the primary
  names;
* the legacy ``NOUS_API_KEY`` / ``NOUS_BASE_URL`` remain as backward-compatible
  fallbacks (so existing deployments keep working);
* when both a generic and a legacy name are set, the generic one wins;
* the model/dimension knobs and the other ``CURRICULUM_*`` settings are
  unchanged.
"""
from __future__ import annotations

import unittest

from curriculum.config import Settings, load


class DefaultsTest(unittest.TestCase):
    def test_empty_env_yields_documented_defaults(self) -> None:
        settings = load({})
        self.assertIsNone(settings.api_key)
        self.assertEqual(
            settings.base_url, "https://inference-api.nousresearch.com/v1"
        )
        self.assertEqual(settings.ingest_model, "deepseek/deepseek-v4-flash")
        self.assertEqual(settings.embed_model, "google/gemini-embedding-2")
        self.assertEqual(settings.embedding_dim, 3072)


class GenericPrimaryTest(unittest.TestCase):
    def test_generic_names_are_read(self) -> None:
        settings = load(
            {
                "CURRICULUM_API_KEY": "generic-key",
                "CURRICULUM_BASE_URL": "https://integrate.api.nvidia.com/v1",
            }
        )
        self.assertEqual(settings.api_key, "generic-key")
        self.assertEqual(settings.base_url, "https://integrate.api.nvidia.com/v1")


class LegacyFallbackTest(unittest.TestCase):
    """Acceptance criterion 1: NOUS_* continues to work."""

    def test_legacy_api_key_falls_back(self) -> None:
        settings = load({"NOUS_API_KEY": "legacy-key"})
        self.assertEqual(settings.api_key, "legacy-key")

    def test_legacy_base_url_falls_back(self) -> None:
        settings = load({"NOUS_BASE_URL": "https://legacy.example.com/v1"})
        self.assertEqual(settings.base_url, "https://legacy.example.com/v1")


class PrecedenceTest(unittest.TestCase):
    """Acceptance criterion 2: generic wins when both are set."""

    def test_generic_api_key_wins_over_legacy(self) -> None:
        settings = load(
            {"CURRICULUM_API_KEY": "generic", "NOUS_API_KEY": "legacy"}
        )
        self.assertEqual(settings.api_key, "generic")

    def test_generic_base_url_wins_over_legacy(self) -> None:
        settings = load(
            {
                "CURRICULUM_BASE_URL": "https://generic.example.com/v1",
                "NOUS_BASE_URL": "https://legacy.example.com/v1",
            }
        )
        self.assertEqual(settings.base_url, "https://generic.example.com/v1")


class UnchangedKnobsTest(unittest.TestCase):
    def test_model_and_dim_env_names_are_unchanged(self) -> None:
        settings = load(
            {
                "CURRICULUM_INGEST_MODEL": "vendor/model-a",
                "CURRICULUM_EMBED_MODEL": "vendor/embed-b",
                "CURRICULUM_EMBED_DIM": "1024",
            }
        )
        self.assertEqual(settings.ingest_model, "vendor/model-a")
        self.assertEqual(settings.embed_model, "vendor/embed-b")
        self.assertEqual(settings.embedding_dim, 1024)

    def test_other_curriculum_settings_still_load(self) -> None:
        settings = load(
            {
                "CURRICULUM_DB_URL": "postgresql://x@localhost/y",
                "CURRICULUM_OKF_PATH": "/tmp/bundle",
                "CURRICULUM_COURSE": "Networking",
            }
        )
        self.assertEqual(settings.database_url, "postgresql://x@localhost/y")
        self.assertEqual(settings.okf_bundle_path, "/tmp/bundle")
        self.assertEqual(settings.default_course, "Networking")

    def test_load_returns_settings_instance(self) -> None:
        self.assertIsInstance(load({}), Settings)


if __name__ == "__main__":
    unittest.main()

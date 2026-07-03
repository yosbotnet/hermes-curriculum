"""Backward-compatible shim for the former Nous-specific provider module.

The concrete implementation moved to
:mod:`curriculum.providers_openai_compatible` when the provider was generalised
to any OpenAI-compatible endpoint. This module is kept as a thin alias so
existing imports (``from curriculum.providers_nous import NousLlm, NousEmbedder``)
keep working unchanged. Prefer importing from
:mod:`curriculum.providers_openai_compatible` in new code.
"""
from __future__ import annotations

from .providers_openai_compatible import (
    OpenAICompatibleEmbedder,
    OpenAICompatibleLlm,
)

# Legacy public names -> the moved implementation.
NousLlm = OpenAICompatibleLlm
NousEmbedder = OpenAICompatibleEmbedder

__all__ = ["NousLlm", "NousEmbedder"]

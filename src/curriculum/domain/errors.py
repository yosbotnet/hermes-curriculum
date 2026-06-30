"""Domain-level exceptions. Infrastructure errors (DB, network) are translated
into these at the adapter boundary so the core never imports driver exceptions."""
from __future__ import annotations


class CurriculumError(Exception):
    """Base for all errors raised by the curriculum layer."""


class ConceptNotFound(CurriculumError):
    pass


class ContentNotFound(CurriculumError):
    pass


class QuestionNotFound(CurriculumError):
    pass


class NoCandidatesAvailable(CurriculumError):
    """next() was asked for an action but nothing is due or learnable."""


class SyncError(CurriculumError):
    pass


class ConfigError(CurriculumError):
    pass

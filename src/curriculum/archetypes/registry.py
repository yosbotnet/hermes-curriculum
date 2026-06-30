"""Course archetypes: named teaching-strategy templates + their registry.

An archetype maps a frozen ``CourseProfile`` to an ``EngineConfig`` that
parameterises the scheduler engine for one course. The menu is bounded and
researched (see docs/learning-engine/hermes-curriculum-layer-design.md Section
4): the model indexes into an evidence base rather than freelancing pedagogy.

Each archetype encodes WHICH learning-science levers matter for its exam shape
through the per-term weights consumed by the ``SelectionPolicy``. The five
weight keys mirror the ``engine.scoring`` term names exactly:

    urgency             -- pay down retention debt (FSRS).
    difficulty_fit      -- aim for the ~85% desirable-difficulty band.
    exploration         -- UCB bonus that keeps the long tail from starving.
    interleave_penalty  -- discourage back-to-back same-cluster confusables;
                           a HIGHER weight means stronger interleaving, because
                           the policy subtracts this term to spread picks out.
    coverage            -- staleness + deadline pressure.

The archetype owns DEFAULT weights and the structural engine toggles; the
per-course ``CourseProfile`` (the decided-once, possibly user-tuned artifact)
gets the final say. Profile weights win on a per-key merge, and
``profile.target_retention`` overrides the archetype -- so a course can nudge
one lever without restating the whole vector, while still inheriting defensible
defaults for everything it leaves alone.

Standard library only; ``engine_config`` is pure (no clock, no RNG, no I/O).
"""
from __future__ import annotations

from typing import Mapping

from ..domain.entities import CourseProfile, EngineConfig
from ..domain.errors import ConfigError
from ..ports.strategies import CourseArchetype

# Canonical weight keys. Defined once (not as bare literals scattered across the
# subclasses) so the vocabulary stays in one place and is greppable against the
# ScoringTerm.name values the SelectionPolicy looks them up by.
WEIGHT_URGENCY = "urgency"
WEIGHT_DIFFICULTY_FIT = "difficulty_fit"
WEIGHT_EXPLORATION = "exploration"
WEIGHT_INTERLEAVE_PENALTY = "interleave_penalty"
WEIGHT_COVERAGE = "coverage"


class _BaseArchetype(CourseArchetype):
    """Shared ``engine_config`` machinery for the concrete archetypes.

    Concrete archetypes are pure data: they set ``DEFAULT_WEIGHTS`` and the three
    engine toggles as class attributes and inherit one merge implementation.
    That split is the Open/Closed seam -- adding an archetype is a new subclass
    with new constants, never an edit to the merge logic that consumes them.
    """

    name: str = "abstract"

    # Per-term default weights for this archetype (overridable per course).
    DEFAULT_WEIGHTS: Mapping[str, float] = {}
    # Structural engine decisions. These are properties of the TEACHING SHAPE,
    # not per-course knobs, so they are not overridable from a CourseProfile.
    ENABLE_FIRE: bool = True
    ENABLE_INTERLEAVE: bool = True
    BASE_TEMPERATURE: float = 0.6

    def engine_config(self, profile: CourseProfile) -> EngineConfig:
        """Resolve this archetype's defaults against a course profile.

        WHY this merge order: the archetype is the researched default; the
        profile is the decided-once (and possibly user-tuned) authority, so the
        profile must win where it speaks. Weights merge per key -- an empty
        ``profile.weights`` leaves the archetype defaults untouched, while a
        non-empty one overrides only the keys it names (and may add new ones).
        ``target_retention`` is a scalar the profile always carries, so it always
        overrides: how hard to study is a course-level decision. The structural
        toggles come straight from the archetype.

        A fresh dict is built every call so the returned ``EngineConfig`` never
        aliases the shared class-level defaults -- a caller mutating the result
        cannot corrupt the archetype or any later call.
        """
        weights: dict[str, float] = dict(self.DEFAULT_WEIGHTS)
        if profile.weights:
            weights.update(profile.weights)
        return EngineConfig(
            weights=weights,
            target_retention=profile.target_retention,
            enable_fire=self.ENABLE_FIRE,
            enable_interleave=self.ENABLE_INTERLEAVE,
            base_temperature=self.BASE_TEMPERATURE,
        )


class ConceptualWrittenArchetype(_BaseArchetype):
    """Conceptual written exams (e.g. Cybersecurity).

    Rewards connections, completeness, terminology, critical discussion. Levers:
    urgency and coverage lead (keep the whole web fresh and complete for a
    long-form exam), while a REAL exploration + interleave presence is what
    surfaces and contrasts cross-topic connections -- exactly what this exam
    shape rewards. FIRe is on (encompass-credit suits a connected web) and
    interleaving is on (confusable topics must be told apart).
    """

    name: str = "conceptual-written"
    DEFAULT_WEIGHTS = {
        WEIGHT_URGENCY: 1.0,
        WEIGHT_DIFFICULTY_FIT: 0.7,
        WEIGHT_EXPLORATION: 0.8,
        WEIGHT_INTERLEAVE_PENALTY: 0.8,
        WEIGHT_COVERAGE: 1.0,
    }
    ENABLE_FIRE = True
    ENABLE_INTERLEAVE = True
    BASE_TEMPERATURE = 0.6


class ProceduralArchetype(_BaseArchetype):
    """Procedural skill exams (math, physics, signals).

    Rewards fluent, error-free execution along a hierarchical prerequisite
    ladder. Levers: urgency + difficulty_fit lead (drill many reps per skill at
    the desirable-difficulty band), and the hierarchy is exploited through FIRe
    (enable_fire) rather than a big coverage/exploration push -- so exploration
    and coverage stay modest. Interleaving is deliberately LOW: a skill ladder
    benefits from blocked practice while a procedure is still being acquired, so
    the interleave_penalty weight is the smallest of any archetype.
    """

    name: str = "procedural"
    DEFAULT_WEIGHTS = {
        WEIGHT_URGENCY: 1.0,
        WEIGHT_DIFFICULTY_FIT: 1.0,
        WEIGHT_EXPLORATION: 0.4,
        WEIGHT_INTERLEAVE_PENALTY: 0.2,
        WEIGHT_COVERAGE: 0.6,
    }
    ENABLE_FIRE = True
    ENABLE_INTERLEAVE = True
    BASE_TEMPERATURE = 0.5


class McqArchetype(_BaseArchetype):
    """Broad-recall / MCQ exams.

    Rewards wide coverage and discrimination. Levers: coverage leads (the
    HIGHEST coverage weight of any archetype -- the exam samples the whole
    syllabus) and exploration is strong (reach the long tail of shallow items),
    while difficulty_fit is low (items are shallow, not deep). FIRe is OFF: there
    is no deep skill hierarchy to propagate credit through, so implicit credit
    would be noise. Spacing + interleaving carry the schedule.
    """

    name: str = "mcq"
    DEFAULT_WEIGHTS = {
        WEIGHT_URGENCY: 0.7,
        WEIGHT_DIFFICULTY_FIT: 0.4,
        WEIGHT_EXPLORATION: 1.0,
        WEIGHT_INTERLEAVE_PENALTY: 0.7,
        WEIGHT_COVERAGE: 1.4,
    }
    ENABLE_FIRE = False
    ENABLE_INTERLEAVE = True
    BASE_TEMPERATURE = 0.7


class VivaArchetype(_BaseArchetype):
    """Oral / viva exams.

    Rewards on-the-spot verbal explanation and handling follow-ups. Levers:
    urgency + difficulty_fit lead -- a viva probes what you can explain fluently
    right now, so keep recall high (urgency) and rehearse at an
    effortful-but-makeable band (difficulty_fit). Interleaving is modest (mix
    topics so follow-ups can jump around, but explanation fluency matters more
    than fine discrimination drills). FIRe is off: verbal fluency is not a
    propagated-credit hierarchy.
    """

    name: str = "viva"
    DEFAULT_WEIGHTS = {
        WEIGHT_URGENCY: 1.0,
        WEIGHT_DIFFICULTY_FIT: 1.0,
        WEIGHT_EXPLORATION: 0.6,
        WEIGHT_INTERLEAVE_PENALTY: 0.5,
        WEIGHT_COVERAGE: 0.7,
    }
    ENABLE_FIRE = False
    ENABLE_INTERLEAVE = True
    BASE_TEMPERATURE = 0.6


class MemorizationArchetype(_BaseArchetype):
    """Memorization-heavy exams (anatomy, law articles).

    Rewards precise recall of a large, mostly flat set. Levers: spacing is king,
    so urgency + coverage are BOTH high (keep a big set from decaying, and keep
    cycling through all of it). The prerequisite structure is weak, so
    difficulty_fit is low. Review is cumulative and interleaved (a moderate
    interleave_penalty). FIRe is OFF -- a flat set has no encompass hierarchy to
    send credit down.
    """

    name: str = "memorization"
    DEFAULT_WEIGHTS = {
        WEIGHT_URGENCY: 1.2,
        WEIGHT_DIFFICULTY_FIT: 0.4,
        WEIGHT_EXPLORATION: 0.6,
        WEIGHT_INTERLEAVE_PENALTY: 0.6,
        WEIGHT_COVERAGE: 1.2,
    }
    ENABLE_FIRE = False
    ENABLE_INTERLEAVE = True
    BASE_TEMPERATURE = 0.5


class CourseArchetypeRegistry:
    """A ``name -> CourseArchetype`` lookup with Open/Closed registration.

    New archetypes are added with ``register`` without editing any consumer;
    ``get`` is the single read path and fails LOUD (``ConfigError``) on an
    unknown name, so a typo'd or unsupported ``profile.archetype`` never silently
    degrades to some default behaviour.
    """

    def __init__(self) -> None:
        self._by_name: dict[str, CourseArchetype] = {}

    def register(self, archetype: CourseArchetype) -> None:
        """Add (or replace) an archetype, keyed by its ``name``.

        Replacing an existing name is allowed on purpose: it lets a deployment
        swap in a re-tuned implementation under the same name without touching
        any call site (the read path is unchanged).
        """
        self._by_name[archetype.name] = archetype

    def get(self, name: str) -> CourseArchetype:
        """Return the archetype registered under ``name``.

        Raises ``ConfigError`` (a domain error, not a bare ``KeyError``) on an
        unknown name so the failure crosses the boundary as a curriculum-layer
        concern with an actionable message listing the known archetypes.
        """
        try:
            return self._by_name[name]
        except KeyError:
            known = ", ".join(sorted(self._by_name)) or "<none>"
            raise ConfigError(
                f"unknown course archetype {name!r}; known archetypes: {known}"
            ) from None

    def names(self) -> tuple[str, ...]:
        """The registered archetype names, sorted (for diagnostics / UX)."""
        return tuple(sorted(self._by_name))


def _build_default_registry() -> CourseArchetypeRegistry:
    """Construct a registry pre-loaded with the five researched archetypes.

    Done in a function (not inline at import) so the construction order is
    explicit and a test can build its own pristine registry the same way.
    """
    registry = CourseArchetypeRegistry()
    for archetype in (
        ConceptualWrittenArchetype(),
        ProceduralArchetype(),
        McqArchetype(),
        VivaArchetype(),
        MemorizationArchetype(),
    ):
        registry.register(archetype)
    return registry


# The process-wide default registry. Importing this module is enough to resolve
# any of the five archetypes by name.
REGISTRY = _build_default_registry()


def register(archetype: CourseArchetype) -> None:
    """Register ``archetype`` in the process-wide default registry."""
    REGISTRY.register(archetype)


def get(name: str) -> CourseArchetype:
    """Look ``name`` up in the default registry (``ConfigError`` if unknown)."""
    return REGISTRY.get(name)


def engine_config_for(profile: CourseProfile) -> EngineConfig:
    """Resolve a profile to its ``EngineConfig`` via the default registry.

    This is the natural ``resolve_config`` callable the application service
    expects (``Callable[[CourseProfile], EngineConfig]``): look the archetype up
    by ``profile.archetype`` and let it merge the profile in. Defined here so the
    wiring lives next to the registry instead of being duplicated at every call
    site, and so an unknown archetype surfaces as a ``ConfigError`` exactly where
    the engine is configured.
    """
    return get(profile.archetype).engine_config(profile)

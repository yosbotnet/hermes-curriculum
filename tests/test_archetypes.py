"""Tests for the course archetypes and their registry (archetypes.registry).

The archetype is the "decided-once" strategy template that turns a CourseProfile
into an EngineConfig. These tests pin down the contract the rest of the engine
relies on:

* the registry resolves each of the five archetypes by name, and fails LOUD
  (ConfigError) on an unknown name -- never a silent default;
* registration is Open/Closed -- a brand-new archetype is reachable through the
  same read path without editing the registry;
* the five enable_fire flags match the documented teaching shapes;
* the per-term weights differ in the documented DIRECTION (e.g. mcq coverage >
  procedural coverage; procedural interleave < conceptual-written interleave);
* a course profile overrides the archetype: non-empty weights merge per key with
  the profile winning, an empty weights map leaves defaults intact, and
  profile.target_retention always overrides.

engine_config is a pure function -- no clock, no RNG -- so every assertion below
is deterministic.
"""
from __future__ import annotations

import unittest

from curriculum.domain.entities import CourseProfile, EngineConfig
from curriculum.domain.errors import ConfigError
from curriculum.ports.strategies import CourseArchetype
from curriculum.archetypes.registry import (
    REGISTRY,
    ConceptualWrittenArchetype,
    CourseArchetypeRegistry,
    McqArchetype,
    MemorizationArchetype,
    ProceduralArchetype,
    VivaArchetype,
    engine_config_for,
    get,
    register,
)

# The five weight keys every archetype must populate (mirrors engine.scoring).
WEIGHT_KEYS = {
    "urgency",
    "difficulty_fit",
    "exploration",
    "interleave_penalty",
    "coverage",
}

# name -> (class, expected enable_fire). The single source of truth the
# parametrised tests below iterate over.
EXPECTED = {
    "conceptual-written": (ConceptualWrittenArchetype, True),
    "procedural": (ProceduralArchetype, True),
    "mcq": (McqArchetype, False),
    "viva": (VivaArchetype, False),
    "memorization": (MemorizationArchetype, False),
}


def profile_for(name: str, **kwargs) -> CourseProfile:
    """A CourseProfile naming an archetype, overriding only what a test cares
    about. Keeps each test focused on one lever of the merge."""
    return CourseProfile(course="c1", archetype=name, **kwargs)


class RegistryLookupTests(unittest.TestCase):
    """get()/registry resolution and the loud-failure invariant."""

    def test_get_returns_each_archetype_by_name(self) -> None:
        for name, (cls, _fire) in EXPECTED.items():
            arch = get(name)
            self.assertIsInstance(arch, cls)
            self.assertEqual(arch.name, name)

    def test_every_archetype_is_a_course_archetype(self) -> None:
        # Liskov: anything the registry hands back must satisfy the port.
        for name in EXPECTED:
            self.assertIsInstance(get(name), CourseArchetype)

    def test_default_registry_lists_exactly_the_five(self) -> None:
        self.assertEqual(set(REGISTRY.names()), set(EXPECTED))

    def test_unknown_name_raises_config_error(self) -> None:
        with self.assertRaises(ConfigError):
            get("no-such-archetype")

    def test_unknown_name_message_lists_known_archetypes(self) -> None:
        with self.assertRaises(ConfigError) as cm:
            get("bogus")
        msg = str(cm.exception)
        self.assertIn("bogus", msg)
        self.assertIn("conceptual-written", msg)

    def test_fresh_registry_is_empty_and_raises(self) -> None:
        empty = CourseArchetypeRegistry()
        self.assertEqual(empty.names(), ())
        with self.assertRaises(ConfigError):
            empty.get("conceptual-written")


class RegistryRegistrationTests(unittest.TestCase):
    """Open/Closed: register() extends the menu without touching the read path."""

    def test_register_then_get_on_isolated_registry(self) -> None:
        reg = CourseArchetypeRegistry()
        arch = ConceptualWrittenArchetype()
        reg.register(arch)
        self.assertIs(reg.get("conceptual-written"), arch)

    def test_register_replaces_same_name(self) -> None:
        reg = CourseArchetypeRegistry()
        first = ProceduralArchetype()
        second = ProceduralArchetype()
        reg.register(first)
        reg.register(second)
        # Same name -> later registration wins (documented swap behaviour).
        self.assertIs(reg.get("procedural"), second)
        self.assertEqual(reg.names(), ("procedural",))

    def test_register_custom_archetype_is_reachable(self) -> None:
        # A brand-new archetype with a new name plugs in with no edits to the
        # registry class -- the Open/Closed promise.
        class FlashcardArchetype(ConceptualWrittenArchetype):
            name = "flashcard-custom"

        reg = CourseArchetypeRegistry()
        custom = FlashcardArchetype()
        reg.register(custom)
        self.assertIs(reg.get("flashcard-custom"), custom)

    def test_module_level_register_adds_to_default_registry(self) -> None:
        class TempArchetype(McqArchetype):
            name = "temp-module-level"

        try:
            register(TempArchetype())
            self.assertIsInstance(get("temp-module-level"), TempArchetype)
        finally:
            # Keep the shared default registry pristine for other test modules.
            REGISTRY._by_name.pop("temp-module-level", None)


class EngineConfigShapeTests(unittest.TestCase):
    """Each archetype produces a well-formed EngineConfig."""

    def test_all_five_weight_keys_present(self) -> None:
        for name in EXPECTED:
            cfg = get(name).engine_config(profile_for(name))
            self.assertEqual(set(cfg.weights), WEIGHT_KEYS, f"{name} weight keys")

    def test_weights_are_positive(self) -> None:
        for name in EXPECTED:
            cfg = get(name).engine_config(profile_for(name))
            for key, value in cfg.weights.items():
                self.assertGreater(value, 0.0, f"{name}.{key}")

    def test_base_temperature_in_unit_range(self) -> None:
        for name in EXPECTED:
            cfg = get(name).engine_config(profile_for(name))
            self.assertGreater(cfg.base_temperature, 0.0, name)
            self.assertLessEqual(cfg.base_temperature, 1.0, name)

    def test_returns_engine_config_instance(self) -> None:
        cfg = ConceptualWrittenArchetype().engine_config(profile_for("conceptual-written"))
        self.assertIsInstance(cfg, EngineConfig)


class EnableFireFlagTests(unittest.TestCase):
    """The structural FIRe/interleave toggles match the documented shapes."""

    def test_enable_fire_flags(self) -> None:
        for name, (_cls, fire) in EXPECTED.items():
            cfg = get(name).engine_config(profile_for(name))
            self.assertEqual(cfg.enable_fire, fire, f"{name} enable_fire")

    def test_conceptual_written_enables_fire_and_interleave(self) -> None:
        cfg = ConceptualWrittenArchetype().engine_config(profile_for("conceptual-written"))
        self.assertTrue(cfg.enable_fire)
        self.assertTrue(cfg.enable_interleave)

    def test_mcq_disables_fire(self) -> None:
        cfg = McqArchetype().engine_config(profile_for("mcq"))
        self.assertFalse(cfg.enable_fire)

    def test_memorization_disables_fire(self) -> None:
        cfg = MemorizationArchetype().engine_config(profile_for("memorization"))
        self.assertFalse(cfg.enable_fire)


class WeightDirectionTests(unittest.TestCase):
    """The weights differ across archetypes in the DOCUMENTED direction.

    Absolute magnitudes are tuning choices; what the contract pins is the
    relative ordering each archetype's pedagogy implies.
    """

    def _w(self, name: str) -> dict:
        return dict(get(name).engine_config(profile_for(name)).weights)

    def test_mcq_coverage_beats_procedural_coverage(self) -> None:
        # Broad recall samples the whole syllabus; a skill ladder does not.
        self.assertGreater(self._w("mcq")["coverage"], self._w("procedural")["coverage"])

    def test_procedural_interleave_below_conceptual_written(self) -> None:
        # Blocked practice suits an acquisition ladder; a connected web wants
        # interleaving to contrast confusable topics.
        self.assertLess(
            self._w("procedural")["interleave_penalty"],
            self._w("conceptual-written")["interleave_penalty"],
        )

    def test_conceptual_explores_more_than_procedural(self) -> None:
        # Connections reward reaching across the graph; the ladder stays focused.
        self.assertGreater(
            self._w("conceptual-written")["exploration"],
            self._w("procedural")["exploration"],
        )

    def test_mcq_difficulty_fit_below_viva(self) -> None:
        # MCQ items are shallow; a viva probes effortful explanation.
        self.assertLess(
            self._w("mcq")["difficulty_fit"],
            self._w("viva")["difficulty_fit"],
        )

    def test_memorization_urgency_and_coverage_are_high(self) -> None:
        # Spacing is king: keep a large flat set from decaying and keep cycling.
        mem = self._w("memorization")
        proc = self._w("procedural")
        self.assertGreater(mem["urgency"], proc["urgency"])
        self.assertGreater(mem["coverage"], proc["coverage"])

    def test_mcq_coverage_is_the_highest(self) -> None:
        coverages = {n: self._w(n)["coverage"] for n in EXPECTED}
        self.assertEqual(max(coverages, key=coverages.get), "mcq")


class ProfileOverrideTests(unittest.TestCase):
    """profile.weights / profile.target_retention override archetype defaults."""

    def test_empty_profile_weights_keeps_archetype_defaults(self) -> None:
        arch = ProceduralArchetype()
        cfg = arch.engine_config(profile_for("procedural"))  # weights default {}
        self.assertEqual(dict(cfg.weights), dict(arch.DEFAULT_WEIGHTS))

    def test_profile_weights_override_named_keys_only(self) -> None:
        arch = ProceduralArchetype()
        profile = profile_for("procedural", weights={"urgency": 5.0, "coverage": 9.0})
        cfg = arch.engine_config(profile)
        # Overridden keys take the profile value...
        self.assertEqual(cfg.weights["urgency"], 5.0)
        self.assertEqual(cfg.weights["coverage"], 9.0)
        # ...while untouched keys retain the archetype default.
        self.assertEqual(cfg.weights["difficulty_fit"], arch.DEFAULT_WEIGHTS["difficulty_fit"])
        self.assertEqual(cfg.weights["interleave_penalty"], arch.DEFAULT_WEIGHTS["interleave_penalty"])

    def test_profile_weights_can_add_new_keys(self) -> None:
        arch = McqArchetype()
        profile = profile_for("mcq", weights={"novel_term": 3.0})
        cfg = arch.engine_config(profile)
        self.assertEqual(cfg.weights["novel_term"], 3.0)
        # Existing defaults survive the merge.
        self.assertEqual(cfg.weights["coverage"], arch.DEFAULT_WEIGHTS["coverage"])

    def test_target_retention_overrides(self) -> None:
        for name in EXPECTED:
            cfg = get(name).engine_config(profile_for(name, target_retention=0.75))
            self.assertEqual(cfg.target_retention, 0.75, name)

    def test_default_target_retention_flows_through(self) -> None:
        # CourseProfile's own default (0.90) is what an unset profile carries.
        cfg = VivaArchetype().engine_config(profile_for("viva"))
        self.assertEqual(cfg.target_retention, 0.90)

    def test_returned_weights_are_an_independent_copy(self) -> None:
        arch = ConceptualWrittenArchetype()
        profile = profile_for("conceptual-written")
        cfg1 = arch.engine_config(profile)
        cfg1.weights["urgency"] = 999.0  # mutate the returned mapping
        # A subsequent call must be unaffected: defaults were copied, not aliased.
        cfg2 = arch.engine_config(profile)
        self.assertNotEqual(cfg2.weights["urgency"], 999.0)
        self.assertEqual(cfg2.weights["urgency"], arch.DEFAULT_WEIGHTS["urgency"])
        self.assertEqual(arch.DEFAULT_WEIGHTS["urgency"], 1.0)


class EngineConfigForTests(unittest.TestCase):
    """engine_config_for is the resolve_config callable the service wants."""

    def test_resolves_via_registry_matches_direct_call(self) -> None:
        profile = profile_for("procedural", weights={"urgency": 2.0}, target_retention=0.8)
        via_helper = engine_config_for(profile)
        direct = get("procedural").engine_config(profile)
        self.assertEqual(dict(via_helper.weights), dict(direct.weights))
        self.assertEqual(via_helper.target_retention, direct.target_retention)
        self.assertEqual(via_helper.enable_fire, direct.enable_fire)
        self.assertEqual(via_helper.base_temperature, direct.base_temperature)

    def test_unknown_archetype_raises_config_error(self) -> None:
        with self.assertRaises(ConfigError):
            engine_config_for(profile_for("not-a-real-archetype"))


if __name__ == "__main__":
    unittest.main()

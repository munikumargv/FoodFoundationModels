import pytest

from foodfm.knowledge import MICRO_STAGES, load_kb, micro_stage

COUNTRY_KEYS = {"id", "name", "region", "guideline", "staples", "typical_meals", "healthy_traditions",
                "concerns", "cultural_religious", "nutrient_sources", "healthier_swaps"}


@pytest.fixture(scope="module")
def kb():
    return load_kb()


def test_has_25_countries_with_complete_entries(kb):
    assert len(kb.countries) == 25
    assert len({c["id"] for c in kb.countries}) == 25
    for c in kb.countries:
        assert COUNTRY_KEYS <= c.keys(), c["id"]
        assert set(c["nutrient_sources"]) == {"protein", "iron", "calcium", "b12"}, c["id"]


def test_no_list_items_split_by_yaml_commas(kb):
    # Unquoted "millets (ragi, jowar)" in a YAML flow list splits into broken items.
    for c in kb.countries:
        items = c["staples"] + c["healthy_traditions"] + [f for v in c["nutrient_sources"].values() for f in v]
        for item in items:
            assert item.count("(") == item.count(")"), (c["id"], item)


def test_age_groups_cover_full_lifespan(kb):
    for age in [0, 0.5, 1, 3.9, 4, 8, 9, 13, 14, 18, 19, 50, 51, 70, 71, 100]:
        assert kb.age_group(age)


def test_micronutrient_tables_cover_all_stages(kb):
    for name, spec in kb.standards["micronutrients"].items():
        for stage in MICRO_STAGES[2:] + ["pregnancy", "lactation"]:
            assert stage in spec["values"], (name, stage)


def test_micro_stage_boundaries():
    assert micro_stage(0.2) == "0-6 months"
    assert micro_stage(0.7) == "7-12 months"
    assert micro_stage(13.9) == "9-13"
    assert micro_stage(14) == "14-18"
    assert micro_stage(71) == "71+"


@pytest.mark.parametrize("age,sex,stage,nutrient,expected", [
    (15, "female", "none", "calcium_mg", 1300),
    (30, "female", "none", "iron_mg", 18),
    (30, "male", "none", "iron_mg", 8),
    (60, "female", "none", "calcium_mg", 1200),
    (75, "male", "none", "vitamin_d_iu", 800),
    (28, "female", "pregnant_t2", "iron_mg", 27),
    (28, "female", "pregnant_t1", "folate_mcg_dfe", 600),
    (17, "female", "pregnant_t3", "calcium_mg", 1300),   # adolescents keep the higher calcium value
    (30, "female", "lactating", "iodine_mcg", 290),
])
def test_micronutrient_reference_values(kb, age, sex, stage, nutrient, expected):
    assert kb.micronutrient_targets(age, sex, stage)[nutrient]["value"] == expected


def test_country_lookup_by_id_or_name(kb):
    assert kb.country("India")["id"] == "india"
    assert kb.country("united states")["id"] == "united_states"
    with pytest.raises(KeyError):
        kb.country("atlantis")


def test_models_config(kb):
    assert kb.models["default"] in kb.models["models"]
    assert kb.model()["hf_id"]
    assert kb.model("org/custom-model")["hf_id"] == "org/custom-model"

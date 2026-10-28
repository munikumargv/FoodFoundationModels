import argparse

import pytest

from foodfm.cli import cmd_analyze
from foodfm.gaps import NUTRIENTS, analyze, analyze_persona, detect_foods, food_allowed
from foodfm.knowledge import load_kb
from foodfm.labs import interpret, parse_lab_text
from foodfm.profile import Profile

NUTRIENT_KEYS = {"kcal", "protein", "carbs", "fat", "sat_fat", "fibre", "free_sugar", "sodium", "calcium", "iron", "zinc",
                 "vit_a", "vit_c", "vit_d", "b12", "folate"}
ALLERGENS = {"milk", "egg", "fish_shellfish", "wheat", "peanut", "tree_nut", "soy", "sesame"}


@pytest.fixture(scope="module")
def kb():
    return load_kb()


def meal(*items):
    return [{"name": "Day", "items": [{"food": f, "servings": s} for f, s in items]}]


# ---------------------------------------------------------------- data integrity
def test_food_table_integrity(kb):
    ids = [f["id"] for f in kb.foods]
    assert len(ids) == len(set(ids)) and len(ids) >= 80
    countries = {c["id"] for c in kb.countries} | {"global"}
    for f in kb.foods:
        assert f["diet"] in {"vegan", "vegetarian", "pescatarian", "meat"}, f["id"]
        assert set(f["allergens"]) <= ALLERGENS, f["id"]
        assert set(f["countries"]) <= countries, f["id"]
        assert set(f["n"]) <= NUTRIENT_KEYS and all(v >= 0 for v in f["n"].values()), f["id"]
        assert f["keywords"] and f["emoji"] and f["serving"], f["id"]


def test_every_country_has_local_foods(kb):
    for c in kb.countries:
        assert sum(c["id"] in f["countries"] for f in kb.foods) >= 2, c["id"]


def test_personas_are_valid(kb):
    for p in kb.personas:
        assert not Profile.from_dict(p["profile"]).validate(kb), p["id"]
        for m in p["meals"]:
            for it in m["items"]:
                kb.food(it["food"])


# ---------------------------------------------------------------- labs
def test_parse_demo_lab_report(kb):
    values = parse_lab_text(kb.personas[0]["lab_report"], kb)
    assert {k: v["value"] for k, v in values.items()} == {
        "hemoglobin": 10.8, "ferritin": 9, "vitamin_d": 14, "vitamin_b12": 182, "hba1c": 5.6, "ldl": 118, "tsh": 2.1}


@pytest.mark.parametrize("line,key,value", [
    ("Vitamin D (25-OH)     26 ng/mL", "vitamin_d", 26),       # the 25 in the name is not the value
    ("Vitamin D, 25-OH 75 nmol/L", "vitamin_d", 30),          # unit conversion
    ("Haemoglobin 118 g/L", "hemoglobin", 11.8),
    ("HbA1c, 6.2, %, 4.0-5.6", "hba1c", 6.2),                 # CSV, and 'hb' must not match HbA1c
    ("Vitamin B12 221 pmol/L", "vitamin_b12", 299.46),
    ("LDL-C 3.4 mmol/L", "ldl", 131.48),
    ("Fasting Blood Sugar 5.9 mmol/L", "fasting_glucose", 106.2),
])
def test_parse_lab_lines(kb, line, key, value):
    parsed = parse_lab_text(line, kb)
    assert set(parsed) == {key}
    assert parsed[key]["value"] == pytest.approx(value, rel=1e-3)


def test_haemoglobin_cutoff_depends_on_sex_and_pregnancy(kb):
    v = {"hemoglobin": {"value": 11.5}}
    woman = Profile(country="india", age=30, sex="female")
    pregnant = Profile(country="india", age=30, sex="female", life_stage="pregnant_t2")
    man = Profile(country="india", age=30, sex="male")
    assert interpret(v, woman, kb)[0]["status"] == "deficient"
    assert interpret(v, pregnant, kb)[0]["status"] == "adequate"
    assert interpret({"hemoglobin": {"value": 12.5}}, man, kb)[0]["status"] == "deficient"


# ---------------------------------------------------------------- analysis
def test_priya_demo(kb):
    r = analyze_persona("priya", kb)
    gaps = {g["key"]: g for g in r["gaps"]}
    for key in ("iron", "b12", "vit_d"):
        assert gaps[key]["status"] == "deficient" and gaps[key]["lab_status"] == "deficient"
    assert gaps["iron"]["target"] == pytest.approx(18 * 1.8)  # vegetarian iron target
    supps = {s["nutrient"]: s for s in r["supplements"]}
    assert supps["iron"]["priority"] == "high" and supps["b12"]["priority"] == "high"
    # Recommendations respect the vegetarian diet and favour Indian foods.
    for f in r["food_recommendations"]:
        assert kb.food(f["food"])["diet"] in ("vegan", "vegetarian")
    assert r["food_recommendations"][0]["local"]
    assert {x["key"] for x in r["excesses"]} == {"sodium"}


def test_james_demo_flags_excess_and_metabolic_labs(kb):
    r = analyze_persona("james", kb)
    assert {x["key"] for x in r["excesses"]} == {"sodium", "free_sugar", "sat_fat"}
    assert "iron" not in {s["nutrient"] for s in r["supplements"]}  # ferritin normal
    assert r["swaps"]


def test_absorption_insight_when_diet_ok_but_lab_low(kb):
    r = analyze_persona("hiroshi", kb)
    b12 = next(n for n in r["nutrients"] if n["key"] == "b12")
    assert b12["pct"] > 100 and b12["status"] == "low" and "absorption" in b12["insight"]


def test_vegan_always_gets_b12(kb):
    p = Profile(country="india", age=30, sex="male", weight_kg=70, height_cm=175, dietary_pattern="vegan")
    r = analyze(p, meal(("fortified_soy_milk", 2), ("dal", 2), ("roti", 4), ("spinach", 1)), kb)
    b12 = next(s for s in r["supplements"] if s["nutrient"] == "b12")
    assert "vegan" in " ".join(b12["reasons"])


def test_high_ferritin_blocks_iron_supplement(kb):
    p = Profile(country="india", age=40, sex="male", weight_kg=70, height_cm=175)
    r = analyze(p, meal(("white_rice", 2)), kb, lab_values={"ferritin": 450})
    assert "iron" not in {s["nutrient"] for s in r["supplements"]}


def test_food_first_nutrients_are_not_supplemented_from_diet_alone(kb):
    p = Profile(country="india", age=40, sex="male", weight_kg=70, height_cm=175)
    r = analyze(p, meal(("white_rice", 2), ("dal", 1)), kb)
    assert {"vit_c", "vit_a"}.isdisjoint({s["nutrient"] for s in r["supplements"]})
    assert any(g["key"] == "vit_c" for g in r["gaps"])


def test_children_get_paediatric_dose_note(kb):
    p = Profile(country="india", age=10, sex="female", weight_kg=30, height_cm=135)
    r = analyze(p, meal(("white_rice", 1)), kb)
    assert r["supplements"] and all("paediatrician" in s["dose"] for s in r["supplements"])


def test_allergies_excluded_from_recommendations(kb):
    p = Profile(country="nigeria", age=30, sex="male", weight_kg=70, height_cm=175, allergies=["peanut", "fish_shellfish"])
    r = analyze(p, meal(("fufu", 2)), kb)
    for f in r["food_recommendations"]:
        assert not {"peanut", "fish_shellfish"} & set(kb.food(f["food"])["allergens"])


def test_food_allowed_rules(kb):
    halal = Profile(country="pakistan", age=30, sex="male", dietary_pattern="halal")
    assert not food_allowed(kb.food("pork"), halal) and food_allowed(kb.food("chicken_grilled"), halal)
    jain = Profile(country="india", age=30, sex="male", dietary_pattern="jain")
    assert not food_allowed(kb.food("potato"), jain) and not food_allowed(kb.food("egg"), jain)


def test_detect_foods_from_photo_name(kb):
    assert [x["food"] for x in detect_foods("IMG_dal_rice_lunch.jpg", "india", kb)][:2] == ["dal", "white_rice"]
    assert detect_foods("IMG_2041.jpg", "india", kb) == []
    assert detect_foods("ugali-with-sukuma.png", "tanzania", kb)[0]["food"] == "maize_porridge"


def test_all_nutrients_reported(kb):
    r = analyze_persona("amara", kb)
    assert {n["key"] for n in r["nutrients"]} == set(NUTRIENTS)
    assert 0 <= r["score"] <= 100


def test_cli_analyze(kb, capsys):
    cmd_analyze(argparse.Namespace(persona="priya", input=None, labs=None, json=False), kb)
    out = capsys.readouterr().out
    assert "NUTRITION GAPS" in out and "SUPPLEMENTS REQUIRED" in out and "FOODS FROM INDIA" in out

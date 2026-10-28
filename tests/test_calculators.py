import pytest

from foodfm.calculators import bmi, bmr, build_targets, energy_target, protein_target, sodium_limit_mg
from foodfm.knowledge import load_kb
from foodfm.profile import Profile


@pytest.fixture(scope="module")
def kb():
    return load_kb()


def adult(**kw):
    base = dict(country="india", age=30, sex="male", weight_kg=70, height_cm=175)
    base.update(kw)
    return Profile(**base)


def test_mifflin_st_jeor():
    # 10*70 + 6.25*175 - 5*30 + 5 = 1648.75
    assert bmr(adult()) == pytest.approx(1648.75)
    assert bmr(adult(sex="female")) == pytest.approx(1648.75 - 166)


def test_schofield_for_children():
    assert bmr(Profile(country="india", age=8, sex="male", weight_kg=25)) == pytest.approx(22.7 * 25 + 495)
    assert bmr(Profile(country="india", age=2, sex="male", weight_kg=12)) is None


def test_energy_with_pal_and_pregnancy(kb):
    e = energy_target(adult(activity_level="moderate"), kb)
    assert e["requirement_kcal"] == round(1648.75 * 1.75)
    preg = adult(sex="female", age=28, life_stage="pregnant_t3")
    assert energy_target(preg, kb)["requirement_kcal"] == round((bmr(preg)) * 1.55 + 452)


def test_weight_loss_deficit_and_floor(kb):
    e = energy_target(adult(goal="weight_loss"), kb)
    assert e["target_kcal"] == e["requirement_kcal"] - 500
    small = Profile(country="japan", age=70, sex="female", weight_kg=40, height_cm=145, activity_level="sedentary", goal="weight_loss")
    assert energy_target(small, kb)["target_kcal"] == 1200


@pytest.mark.parametrize("profile", [
    Profile(country="india", age=12, sex="female", weight_kg=50, height_cm=150, goal="weight_loss"),
    Profile(country="india", age=30, sex="female", weight_kg=70, height_cm=160, life_stage="lactating", goal="weight_loss"),
    Profile(country="india", age=25, sex="female", weight_kg=50, height_cm=165, medical_conditions=["eating_disorder_history"], goal="weight_loss"),
])
def test_no_deficit_for_protected_groups(kb, profile):
    e = energy_target(profile, kb)
    assert e["target_kcal"] == e["requirement_kcal"]


def test_protein_targets(kb):
    assert protein_target(adult(), kb)["g_per_day"] == 56
    assert protein_target(adult(age=70), kb)["g_per_kg"] == 1.0
    assert protein_target(adult(sex="female", life_stage="pregnant_t2"), kb)["g_per_kg"] == 1.1
    assert protein_target(adult(goal="muscle_gain"), kb)["g_per_kg"] == 1.6
    assert protein_target(adult(medical_conditions=["chronic_kidney_disease"]), kb)["g_per_day"] is None
    assert protein_target(Profile(country="india", age=0.3, sex="male", weight_kg=6), kb)["g_per_day"] == 9.1


def test_protein_uses_reference_weight_in_obesity(kb):
    p = adult(weight_kg=120, height_cm=170)
    assert protein_target(p, kb)["g_per_day"] == round(0.8 * 25 * 1.7 ** 2)


def test_sodium_limits():
    assert sodium_limit_mg(Profile(country="india", age=1, sex="male")) is None
    assert sodium_limit_mg(adult()) == 2000
    assert sodium_limit_mg(adult(medical_conditions=["hypertension"])) == 1500
    assert sodium_limit_mg(Profile(country="india", age=6, sex="male")) < 2000


def test_build_targets_shape(kb):
    t = build_targets(adult(dietary_pattern="vegan", lifestyle=["smoker"]), kb)
    assert t["bmi"] == bmi(adult())
    assert t["micronutrients"]["vitamin_c_mg"]["value"] == 90 + 35
    assert "note" in t["micronutrients"]["iron_mg"]
    assert t["macros"]["fat"]["percent_energy"] == [20, 35]

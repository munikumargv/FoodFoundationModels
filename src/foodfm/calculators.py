"""Deterministic nutrition calculations.

Numbers the model states (energy, protein, micronutrients) are computed here
from standards.yaml and injected into the prompt, so the LLM explains and
localises them rather than inventing them.
"""

from __future__ import annotations

from typing import Any

from .knowledge import KnowledgeBase
from .profile import Profile

# FAO/WHO/UNU (Schofield 1985) BMR equations, kcal/day = a * weight_kg + b.
_SCHOFIELD = {
    "male": [(3, 10, 22.7, 495), (10, 18, 17.5, 651)],
    "female": [(3, 10, 22.5, 499), (10, 18, 12.2, 746)],
}

# Conditions / situations where an energy deficit must not be suggested.
_NO_DEFICIT_CONDITIONS = {"eating_disorder_history", "underweight_malnutrition", "chronic_kidney_disease"}


def bmi(p: Profile) -> float | None:
    if not p.weight_kg or not p.height_cm:
        return None
    return round(p.weight_kg / (p.height_cm / 100) ** 2, 1)


def bmi_category(value: float | None, age: float) -> str | None:
    """WHO adult BMI categories. Children need BMI-for-age z-scores (not computed)."""
    if value is None:
        return None
    if age < 18:
        return "use WHO BMI-for-age growth charts for under-18s"
    if value < 18.5:
        return "underweight"
    if value < 25:
        return "normal weight"
    if value < 30:
        return "overweight"
    return "obesity"


def bmr(p: Profile) -> float | None:
    """Basal metabolic rate in kcal/day, or None when it cannot be estimated."""
    if p.weight_kg is None or p.age < 3:
        return None
    if p.age >= 18:
        if p.height_cm is None:
            return None
        base = 10 * p.weight_kg + 6.25 * p.height_cm - 5 * p.age
        return base + (5 if p.sex == "male" else -161)
    for lo, hi, a, b in _SCHOFIELD[p.sex]:
        if lo <= p.age < hi:
            return a * p.weight_kg + b
    return None


def energy_target(p: Profile, kb: KnowledgeBase) -> dict[str, Any]:
    """Estimated energy requirement and goal-adjusted target (kcal/day)."""
    energy = kb.standards["energy"]
    base = bmr(p)
    if base is None:
        reason = (
            "Energy needs for children under 3 follow growth and appetite; follow paediatric guidance."
            if p.age < 3
            else "Weight and height are required to estimate energy needs."
        )
        return {"requirement_kcal": None, "target_kcal": None, "notes": [reason]}

    pal = energy["physical_activity_levels"][p.activity_level]
    requirement = base * pal
    notes = [f"BMR {round(base)} kcal x PAL {pal} ({p.activity_level})"]

    if p.life_stage.startswith("pregnant"):
        trimester = "trimester_" + p.life_stage[-1]
        extra = energy["pregnancy_extra_kcal"][trimester]
        requirement += extra
        notes.append(f"+{extra} kcal for pregnancy ({trimester.replace('_', ' ')})")
    elif p.life_stage == "lactating":
        requirement += energy["lactation_extra_kcal"]
        notes.append(f"+{energy['lactation_extra_kcal']} kcal for lactation")

    target = requirement
    deficit_blocked = (
        p.age < 18
        or p.life_stage != "none"
        or bool(_NO_DEFICIT_CONDITIONS & set(p.medical_conditions))
    )
    if p.goal == "weight_loss":
        if deficit_blocked:
            notes.append("Weight-loss deficit NOT applied (child/adolescent, pregnancy/lactation, or condition requiring clinical supervision).")
        else:
            floor = 1500 if p.sex == "male" else 1200
            target = max(requirement - 500, floor)
            notes.append(f"-500 kcal/day for gradual weight loss (floor {floor} kcal)")
    elif p.goal in ("weight_gain", "muscle_gain"):
        surplus = 400 if p.goal == "weight_gain" else 250
        target = requirement + surplus
        notes.append(f"+{surplus} kcal/day for {p.goal.replace('_', ' ')}")

    return {"requirement_kcal": round(requirement), "target_kcal": round(target), "notes": notes}


def protein_target(p: Profile, kb: KnowledgeBase) -> dict[str, Any]:
    """Protein target in g/day (or g/kg range where weight is unknown)."""
    table = kb.standards["protein_g_per_kg"]
    if p.age < 1:
        key = "0-6 months" if p.age < 0.5 else "7-12 months"
        return {"g_per_day": kb.standards["protein_infant_g_per_day"][key], "g_per_kg": None, "basis": "NASEM AI for infants"}

    if p.life_stage.startswith("pregnant"):
        per_kg, basis = table["pregnancy"], "NASEM RDA, pregnancy"
    elif p.life_stage == "lactating":
        per_kg, basis = table["lactation"], "NASEM RDA, lactation"
    elif p.age < 4:
        per_kg, basis = table["1-3"], "NASEM RDA"
    elif p.age < 14:
        per_kg, basis = table["4-13"], "NASEM RDA"
    elif p.age < 19:
        per_kg, basis = table["14-18"], "NASEM RDA"
    elif p.age < 65:
        per_kg, basis = table["19-64"], "NASEM RDA"
    else:
        per_kg, basis = table["65+"], "PROT-AGE/ESPEN older-adult recommendation"

    if "chronic_kidney_disease" in p.medical_conditions:
        return {"g_per_day": None, "g_per_kg": None, "basis": "CKD: protein must be individualised by a renal dietitian"}
    if p.age >= 18 and p.goal == "muscle_gain":
        per_kg, basis = 1.6, "sports nutrition consensus (1.6-2.2 g/kg) for muscle gain"
    elif p.age >= 18 and p.activity_level == "vigorous":
        per_kg, basis = max(per_kg, 1.4), "sports nutrition (1.4-2.0 g/kg) for vigorous activity"
    elif p.age >= 18 and p.goal == "weight_loss":
        per_kg, basis = max(per_kg, 1.0), "higher protein to preserve lean mass during weight loss"

    weight = p.weight_kg
    b = bmi(p)
    if weight and b is not None and p.age >= 18 and b >= 30:
        # Actual weight overstates protein needs in obesity; use the weight at BMI 25.
        weight = 25 * (p.height_cm / 100) ** 2
        basis += "; based on reference weight at BMI 25"
    g = round(per_kg * weight) if weight else None
    return {"g_per_day": g, "g_per_kg": per_kg, "basis": basis}


def macro_ranges(p: Profile, kb: KnowledgeBase, target_kcal: float | None) -> dict[str, Any]:
    """AMDR percentages and, if energy is known, gram ranges."""
    if p.age < 1:
        return {}
    key = "1-3" if p.age < 4 else "4-18" if p.age < 19 else "19+"
    amdr = kb.standards["amdr"][key]
    kcal_per_g = {"protein": 4, "fat": 9, "carbohydrate": 4}
    out: dict[str, Any] = {}
    for macro, (lo, hi) in amdr.items():
        entry: dict[str, Any] = {"percent_energy": [lo, hi]}
        if target_kcal:
            entry["grams"] = [round(target_kcal * lo / 100 / kcal_per_g[macro]), round(target_kcal * hi / 100 / kcal_per_g[macro])]
        out[macro] = entry
    if target_kcal:
        out["free_sugars_max_g"] = round(target_kcal * 0.10 / 4)
        out["saturated_fat_max_g"] = round(target_kcal * 0.10 / 9)
    return out


def sodium_limit_mg(p: Profile) -> int | None:
    """Daily sodium ceiling in mg.

    WHO sets 2000 mg for adults and says to scale down for children by energy
    need without giving numbers; we scale the NASEM CDRR age bands
    (1200/1500/1800/2300) by WHO/NASEM adult ratio (2000/2300).
    """
    if p.age < 2:
        return None  # no added salt for children under 2
    if p.age < 4:
        limit = 1000
    elif p.age < 9:
        limit = 1300
    elif p.age < 14:
        limit = 1600
    else:
        limit = 2000
    if "hypertension" in p.medical_conditions or "chronic_kidney_disease" in p.medical_conditions:
        limit = min(limit, 1500)
    return limit


def build_targets(p: Profile, kb: KnowledgeBase) -> dict[str, Any]:
    energy = energy_target(p, kb)
    b = bmi(p)
    micros = kb.micronutrient_targets(p.age, p.sex, p.life_stage)
    if "smoker" in p.lifestyle and "vitamin_c_mg" in micros:
        micros["vitamin_c_mg"] = {**micros["vitamin_c_mg"], "value": micros["vitamin_c_mg"]["value"] + 35}
    if p.dietary_pattern in ("vegetarian_lacto_ovo", "vegan", "jain", "buddhist_vegetarian") and "iron_mg" in micros:
        iron = micros["iron_mg"]["value"]
        micros["iron_mg"] = {**micros["iron_mg"], "note": f"plant-based diet: aim for ~{round(iron * 1.8)} mg from food (1.8x, lower non-haem absorption)"}
    return {
        "bmi": b,
        "bmi_category": bmi_category(b, p.age),
        "energy": energy,
        "protein": protein_target(p, kb),
        "macros": macro_ranges(p, kb, energy["target_kcal"]),
        "sodium_max_mg": sodium_limit_mg(p),
        "micronutrients": micros,
    }

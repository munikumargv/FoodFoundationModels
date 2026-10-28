"""Nutrition gap analysis: meals (from photos) + profile + optional labs.

Output: nutrient gaps against the RDA, supplements needed, and food
recommendations from the person's country.
"""

from __future__ import annotations

import re
from typing import Any

from .calculators import build_targets
from .knowledge import KnowledgeBase
from .labs import interpret, parse_lab_text
from .profile import Profile

# Nutrients compared with the RDA: key -> (label, unit, micronutrient-table key)
NUTRIENTS: dict[str, tuple[str, str, str | None]] = {
    "protein": ("Protein", "g", None),
    "fibre": ("Fibre", "g", "fibre_g"),
    "calcium": ("Calcium", "mg", "calcium_mg"),
    "iron": ("Iron", "mg", "iron_mg"),
    "zinc": ("Zinc", "mg", "zinc_mg"),
    "vit_a": ("Vitamin A", "mcg RAE", "vitamin_a_mcg_rae"),
    "vit_c": ("Vitamin C", "mg", "vitamin_c_mg"),
    "vit_d": ("Vitamin D", "IU", "vitamin_d_iu"),
    "b12": ("Vitamin B12", "mcg", "vitamin_b12_mcg"),
    "folate": ("Folate", "mcg DFE", "folate_mcg_dfe"),
}
# Nutrients with an upper limit (WHO): key -> (label, unit)
LIMITS: dict[str, tuple[str, str]] = {
    "sodium": ("Sodium", "mg"),
    "free_sugar": ("Free sugars", "g"),
    "sat_fat": ("Saturated fat", "g"),
}
SEVERITY = {"deficient": 3, "low": 2, "borderline": 1, "adequate": 0}
PLANT_BASED = {"vegetarian_lacto_ovo", "vegan", "jain", "buddhist_vegetarian"}

_DIET_ALLOWED = {
    "vegan": {"vegan"},
    "vegetarian_lacto_ovo": {"vegan", "vegetarian"},
    "jain": {"vegan", "vegetarian"},
    "buddhist_vegetarian": {"vegan", "vegetarian"},
    "pescatarian": {"vegan", "vegetarian", "pescatarian"},
}
_DIET_BANNED_FOODS = {
    "hindu_no_beef": {"beef", "burger", "tacos", "pho"},
    "halal": {"pork"},
    "kosher": {"pork", "shrimp"},
    "jain": {"egg", "potato", "sweet_potato", "carrot", "nasi_goreng", "mandazi"},
}
# Foods never proposed as recommendations (high in sodium/sugar/saturated fat).
_NOT_RECOMMENDED = {"free_sugar": 5, "sodium": 800, "sat_fat": 6}


def food_allowed(food: dict[str, Any], p: Profile) -> bool:
    allowed = _DIET_ALLOWED.get(p.dietary_pattern)
    if allowed is not None and food["diet"] not in allowed:
        return False
    if food["id"] in _DIET_BANNED_FOODS.get(p.dietary_pattern, set()):
        return False
    return not set(food.get("allergens", [])) & set(p.allergies)


def detect_foods(label: str, country: str, kb: KnowledgeBase, limit: int = 4) -> list[dict[str, Any]]:
    """Map a photo label (file name or vision-model caption) to foods in the database.

    Matches food keywords as whole words, in the order they appear in the label;
    when two foods match the same words, the country-typical one wins.
    """
    text = " " + re.sub(r"[^a-z0-9]+", " ", label.lower()) + " "
    hits = []
    for f in kb.foods:
        for k in f["keywords"]:
            i = text.find(f" {k.lower()} ")
            if i >= 0:
                hits.append((i, -len(k), country not in f["countries"], f))
    # Earliest, then longest match first ("brown rice" beats "rice"); spans may not overlap.
    hits.sort(key=lambda h: h[:3])
    chosen: list[dict[str, Any]] = []
    taken: list[tuple[int, int]] = []
    for i, neg_len, _, f in hits:
        span = (i, i - neg_len)
        if any(a < span[1] and span[0] < b for a, b in taken) or f["id"] in {c["food"] for c in chosen}:
            continue
        taken.append(span)
        chosen.append({"food": f["id"], "servings": 1})
    return chosen[:limit]


def daily_intake(meals: list[dict[str, Any]], kb: KnowledgeBase) -> tuple[dict[str, float], dict[str, list[tuple[str, float]]]]:
    """Sum nutrients over all meals; also return the top contributing foods per nutrient."""
    totals: dict[str, float] = {}
    contrib: dict[str, dict[str, float]] = {}
    for meal in meals:
        for item in meal.get("items", []):
            food = kb.food(item["food"])
            servings = float(item.get("servings", 1))
            for k, v in food["n"].items():
                amount = v * servings
                totals[k] = totals.get(k, 0.0) + amount
                contrib.setdefault(k, {})
                contrib[k][food["name"]] = contrib[k].get(food["name"], 0.0) + amount
    top = {k: sorted(c.items(), key=lambda x: -x[1])[:3] for k, c in contrib.items()}
    return totals, top


def _targets(p: Profile, kb: KnowledgeBase) -> tuple[dict[str, float], dict[str, float], dict[str, Any]]:
    t = build_targets(p, kb)
    micros = t["micronutrients"]
    targets: dict[str, float] = {}
    for key, (_, _, micro_key) in NUTRIENTS.items():
        if key == "protein":
            g = t["protein"]["g_per_day"]
            if g:
                targets[key] = g
        elif micro_key in micros:
            targets[key] = micros[micro_key]["value"]
    if p.dietary_pattern in PLANT_BASED and "iron" in targets:
        targets["iron"] = round(targets["iron"] * 1.8, 1)  # lower non-haem iron absorption (NASEM)

    kcal = t["energy"]["target_kcal"] or 2000
    limits = {
        "sodium": t["sodium_max_mg"] or 0,
        "free_sugar": round(kcal * 0.10 / 4),
        "sat_fat": round(kcal * 0.10 / 9),
    }
    return targets, limits, t


def _status(pct: float) -> str:
    if pct < 50:
        return "deficient"
    if pct < 80:
        return "low"
    if pct < 100:
        return "borderline"
    return "adequate"


def _escalate(current: str, lab_status: str) -> str:
    lab_level = {"deficient": "deficient", "low": "low"}.get(lab_status)
    if lab_level and SEVERITY[lab_level] > SEVERITY[current]:
        return lab_level
    return current


def _supplements(p: Profile, kb: KnowledgeBase, rows: dict[str, dict[str, Any]], lab_by_marker: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    blocked = set()
    if lab_by_marker.get("ferritin", {}).get("status") == "excess":
        blocked.add("iron")
    if lab_by_marker.get("calcium", {}).get("status") == "excess":
        blocked |= {"calcium", "vit_d"}
    if lab_by_marker.get("vitamin_d", {}).get("status") == "excess":
        blocked.add("vit_d")
    if "chronic_kidney_disease" in p.medical_conditions:
        blocked.add("protein")

    out = []
    for key, rule in kb.labs["supplements"].items():
        row = rows.get(key)
        if row is None or key in blocked:
            continue
        lab_status = row.get("lab_status")
        reasons, level = [], None
        if lab_status in ("deficient", "low"):
            level = lab_status
            reasons.append(f"Lab: {row['lab_label']} {row['lab_value']:g} {row['lab_unit']} ({lab_status})")
        if row["pct"] < rule["diet_trigger_pct"]:
            level = level or "diet"
            reasons.append(f"Diet provides only {row['pct']:.0f}% of the RDA")
        cond = rule.get("always_if", {})
        always = (
            p.dietary_pattern in cond.get("diet", [])
            or ("age_min" in cond and p.age >= cond["age_min"])
            or p.life_stage in cond.get("life_stage", [])
        )
        if always:
            level = level or "diet"
            if p.dietary_pattern in cond.get("diet", []):
                reasons.append(f"Recommended for all {p.dietary_pattern.replace('_', ' ')} diets")
            elif p.life_stage in cond.get("life_stage", []):
                reasons.append("Recommended during pregnancy")
            else:
                reasons.append(f"Recommended at age {cond['age_min']}+")
        if level is None or (level == "diet" and rule.get("food_first")):
            continue
        # A normal lab result with a modest dietary shortfall: food first, no supplement.
        if lab_status == "adequate" and not always and row["pct"] >= rule["diet_trigger_pct"] / 2:
            continue
        priority = {"deficient": "high", "low": "medium", "diet": "medium" if always else "consider"}[level]
        dose = rule["dose"][level]
        if p.age < 18:
            dose = "Child/teen dose differs - ask your paediatrician for the right amount"
        out.append({"nutrient": key, "name": rule["name"], "form": rule["form"], "dose": dose, "priority": priority,
                    "reasons": reasons, "tips": rule["tips"], "caution": rule["caution"]})
    order = {"high": 0, "medium": 1, "consider": 2}
    out.sort(key=lambda s: order[s["priority"]])
    return out


def _food_recommendations(p: Profile, kb: KnowledgeBase, gaps: list[dict[str, Any]], targets: dict[str, float], limit: int = 8) -> list[dict[str, Any]]:
    candidates = [
        f for f in kb.foods
        if food_allowed(f, p)
        and (p.country in f["countries"] or "global" in f["countries"])
        and all(f["n"].get(k, 0) <= v for k, v in _NOT_RECOMMENDED.items())
    ]
    picks: dict[str, dict[str, Any]] = {}
    for gap in gaps:
        key = gap["key"]
        if key not in targets:
            continue
        scored = []
        for f in candidates:
            amount = f["n"].get(key, 0)
            if amount <= 0:
                continue
            pct = amount / targets[key] * 100
            scored.append((pct * (1.5 if p.country in f["countries"] else 1.0), pct, amount, f))
        scored.sort(key=lambda s: -s[0])
        for _, pct, amount, f in scored[:3]:
            entry = picks.setdefault(f["id"], {"food": f["id"], "name": f["name"], "emoji": f["emoji"], "serving": f["serving"],
                                               "local": p.country in f["countries"], "boosts": [], "_score": 0.0})
            entry["boosts"].append({"nutrient": key, "label": NUTRIENTS[key][0], "amount": round(amount, 1),
                                    "unit": NUTRIENTS[key][1], "pct_rda": round(pct)})
            entry["_score"] += min(pct, 100) * (1 + SEVERITY[gap["status"]])
    ranked = sorted(picks.values(), key=lambda e: (-len(e["boosts"]), -e["_score"]))[:limit]
    for e in ranked:
        e.pop("_score")
    return ranked


def analyze(p: Profile, meals: list[dict[str, Any]], kb: KnowledgeBase, lab_text: str = "",
            lab_values: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run the full gap analysis.

    meals:      [{"name": "Lunch", "items": [{"food": "dal", "servings": 1}, ...]}, ...] for one day
    lab_text:   optional raw lab report text (parsed)
    lab_values: optional {marker: value} overrides (e.g. edited in the UI), canonical units
    """
    errors = p.validate(kb)
    if errors:
        raise ValueError("; ".join(errors))
    intake, top = daily_intake(meals, kb)
    targets, limits, t = _targets(p, kb)

    parsed = parse_lab_text(lab_text, kb) if lab_text else {}
    for key, value in (lab_values or {}).items():
        if value not in (None, "") and key in kb.labs["markers"]:
            parsed[key] = {"value": float(value), "unit": kb.labs["markers"][key]["unit"], "raw": "entered manually"}
    findings = interpret(parsed, p, kb)
    lab_by_marker = {f["marker"]: f for f in findings}
    # Worst lab status per nutrient (e.g. haemoglobin and ferritin both inform iron).
    lab_by_nutrient: dict[str, dict[str, Any]] = {}
    rank = {"deficient": 3, "excess": 3, "low": 2, "adequate": 0}
    for f in findings:
        n = f["nutrient"]
        if n and (n not in lab_by_nutrient or rank[f["status"]] > rank[lab_by_nutrient[n]["status"]]):
            lab_by_nutrient[n] = f

    rows: dict[str, dict[str, Any]] = {}
    for key, (label, unit, _) in NUTRIENTS.items():
        if key not in targets:
            continue
        value = intake.get(key, 0.0)
        pct = value / targets[key] * 100 if targets[key] else 100.0
        diet_status = _status(pct)
        row = {"key": key, "label": label, "unit": unit, "kind": "nutrient", "intake": round(value, 1), "target": targets[key],
               "pct": round(pct, 1), "diet_status": diet_status, "status": diet_status,
               "sources": [{"food": n, "amount": round(a, 1)} for n, a in top.get(key, [])]}
        lab = lab_by_nutrient.get(key)
        if lab:
            row.update(lab_status=lab["status"], lab_label=lab["label"], lab_value=lab["value"], lab_unit=lab["unit"], lab_note=lab["note"])
            row["status"] = _escalate(diet_status, lab["status"])
            if diet_status in ("adequate", "borderline") and lab["status"] in ("deficient", "low"):
                row["insight"] = ("Your meals supply enough, but your blood level is low - this points to poor absorption "
                                  "(common after 50, with metformin/antacids or gut conditions) or higher needs. Discuss with your doctor.")
            elif SEVERITY[diet_status] >= 2 and lab["status"] == "adequate":
                row["insight"] = "Blood level is still normal, but intake is low - stores will run down over time if the diet stays like this."
        rows[key] = row

    limit_rows = []
    for key, (label, unit) in LIMITS.items():
        if not limits.get(key):
            continue
        value = intake.get(key, 0.0)
        pct = value / limits[key] * 100
        status = "excess" if pct > 100 else "adequate"
        row = {"key": key, "label": label, "unit": unit, "kind": "limit", "intake": round(value, 1), "target": limits[key],
               "pct": round(pct, 1), "status": status,
               "sources": [{"food": n, "amount": round(a, 1)} for n, a in top.get(key, [])]}
        lab = lab_by_nutrient.get(key)
        if lab and lab["status"] in ("excess", "low"):
            row.update(lab_status=lab["status"], lab_label=lab["label"], lab_value=lab["value"], lab_unit=lab["unit"], lab_note=lab["note"])
            row["status"] = "excess" if (pct > 100 or lab["status"] == "excess") else "watch"
        limit_rows.append(row)

    gaps = sorted((r for r in rows.values() if r["status"] != "adequate"), key=lambda r: (-SEVERITY[r["status"]], r["pct"]))
    excesses = [r for r in limit_rows if r["status"] in ("excess", "watch")]

    # Score: average RDA coverage (capped at 100%) minus 8 points per limit exceeded.
    coverage = sum(min(r["pct"], 100) for r in rows.values()) / max(len(rows), 1)
    score = max(0, round(coverage - 8 * sum(r["status"] == "excess" for r in limit_rows)))

    c = kb.country(p.country)
    energy_target = t["energy"]["target_kcal"]
    return {
        "profile": p.to_dict(),
        "country": {"id": c["id"], "name": c["name"], "guideline": c["guideline"]},
        "score": score,
        "energy": {"intake": round(intake.get("kcal", 0)), "target": energy_target,
                   "pct": round(intake.get("kcal", 0) / energy_target * 100) if energy_target else None},
        "macros": {k: round(intake.get(k, 0), 1) for k in ("protein", "carbs", "fat")},
        "nutrients": list(rows.values()),
        "limits": limit_rows,
        "gaps": gaps,
        "excesses": excesses,
        "lab_findings": findings,
        "supplements": _supplements(p, kb, rows, lab_by_marker),
        "food_recommendations": _food_recommendations(p, kb, gaps, targets),
        "swaps": c["healthier_swaps"] if excesses else [],
        "country_sources": c["nutrient_sources"],
        "disclaimer": ("Estimates from photo-detected portions and a reference food table; supplement suggestions are "
                       "general guidance based on WHO/NASEM reference values. Confirm with a doctor or registered "
                       "dietitian before starting supplements, especially in pregnancy, childhood or with medical conditions."),
    }


def analyze_persona(persona_id: str, kb: KnowledgeBase) -> dict[str, Any]:
    persona = next(x for x in kb.personas if x["id"] == persona_id)
    return analyze(Profile.from_dict(persona["profile"]), persona["meals"], kb, lab_text=persona.get("lab_report", ""))

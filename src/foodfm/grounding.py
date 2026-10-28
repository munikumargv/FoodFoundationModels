"""Turn a profile + knowledge base into grounded prompts and reference answers."""

from __future__ import annotations

from typing import Any

from .calculators import build_targets
from .knowledge import KnowledgeBase
from .profile import Profile
from .retrieval import BM25, kb_chunks

SYSTEM_PROMPT = """You are FoodFM, a nutrition assistant. Your knowledge is limited to:
- nutrition recommendations according to WHO, FAO and national dietary reference standards,
- food habits of the 25 most populous countries,
- how age, sex, pregnancy/lactation, activity, dietary/religious practice, health conditions, allergies, budget, climate and lifestyle change those recommendations.

Rules:
1. Ground every number in the PROFILE TARGETS or REFERENCE FACTS provided. Never invent values; if a value is missing, say so.
2. Adapt advice to the person's country food habits, using local foods and dishes.
3. Respect dietary, religious and allergy constraints strictly.
4. For medical conditions, infants, pregnancy, eating disorders or medications, give general guidance and tell the person to confirm with the relevant clinician.
5. If a question is not about food or nutrition, politely decline and say you only answer nutrition questions.
6. You are not a substitute for a doctor or registered dietitian."""

OUT_OF_SCOPE_REPLY = (
    "I'm FoodFM, a nutrition assistant, so I can only help with food and nutrition questions "
    "- for example dietary recommendations by age, country food habits, or adapting meals to health "
    "conditions. Is there something about your diet I can help with?"
)

_LABELS = {
    "calcium_mg": "Calcium", "iron_mg": "Iron", "vitamin_d_iu": "Vitamin D", "vitamin_b12_mcg": "Vitamin B12",
    "folate_mcg_dfe": "Folate", "zinc_mg": "Zinc", "iodine_mcg": "Iodine", "vitamin_a_mcg_rae": "Vitamin A",
    "vitamin_c_mg": "Vitamin C", "fibre_g": "Fibre", "water_l": "Total water (food + drinks)",
}


_STAGE_LABELS = {
    "pregnant_t1": "Pregnancy (first trimester)", "pregnant_t2": "Pregnancy (second trimester)",
    "pregnant_t3": "Pregnancy (third trimester)", "lactating": "Breastfeeding",
}


def _humanize(s: str) -> str:
    return s.replace("_", " ")


def _fmt_micro(key: str, v: dict[str, Any]) -> str:
    s = f"{_LABELS.get(key, key)} {v['value']} {v['unit']}"
    return f"{s} ({v['note']})" if v.get("note") else s


def _for_sex(items: list[str], sex: str) -> list[str]:
    """Drop priorities that explicitly address the other sex."""
    other = ("Men", "Boys") if sex == "female" else ("Women", "Girls")
    return [x for x in items if not x.startswith(other)]


def _special_stage(p: Profile, kb: KnowledgeBase) -> dict[str, Any] | None:
    if p.life_stage.startswith("pregnant"):
        return kb.special_life_stages["pregnancy"]
    if p.life_stage == "lactating":
        return kb.special_life_stages["lactation"]
    return None


def _factor_notes(p: Profile, kb: KnowledgeBase) -> tuple[list[str], list[str]]:
    """Collect adjustments and safety notes for every selected factor."""
    adjustments: list[str] = []
    safety: list[str] = []
    # Sex is already reflected in the computed targets, so it is not listed here.
    # Activity, adult diet patterns and goals do not apply to children under 2.
    selected: list[tuple[str, str]] = [("life_stage", p.life_stage), ("budget", p.budget), ("climate", p.climate)]
    if p.age >= 2:
        selected += [("activity_level", p.activity_level), ("dietary_pattern", p.dietary_pattern), ("goal", p.goal)]
    selected += [("medical_conditions", c) for c in p.medical_conditions]
    selected += [("allergies", a) for a in p.allergies]
    selected += [("lifestyle", x) for x in p.lifestyle]
    for fname, opt in selected:
        spec = kb.factors[fname]["options"].get(opt, {})
        adjustments += spec.get("adjustments", [])
        if spec.get("safety"):
            safety.append(spec["safety"])
    if p.allergies and p.allergies != ["none"]:
        safety.append(kb.factors["allergies"]["safety"])
    return adjustments, safety


def profile_context(p: Profile, kb: KnowledgeBase) -> str:
    """Structured, model-readable context block for a profile."""
    t = build_targets(p, kb)
    c = kb.country(p.country)
    g = kb.age_group(p.age)
    adjustments, safety = _factor_notes(p, kb)

    lines = ["PROFILE:"]
    lines.append(
        f"- {c['name']}, age {p.age:g}, {p.sex}, weight {p.weight_kg or 'unknown'} kg, height {p.height_cm or 'unknown'} cm, "
        f"activity {p.activity_level}, life stage {_humanize(p.life_stage)}, diet {_humanize(p.dietary_pattern)}, goal {_humanize(p.goal)}, "
        f"budget {p.budget}, climate {_humanize(p.climate)}"
    )
    lines.append(f"- Conditions: {', '.join(map(_humanize, p.medical_conditions)) or 'none'}; allergies: {', '.join(map(_humanize, p.allergies)) or 'none'}; lifestyle: {', '.join(map(_humanize, p.lifestyle)) or 'none'}")

    lines.append("PROFILE TARGETS (computed from standards):")
    if t["bmi"] is not None:
        lines.append(f"- BMI {t['bmi']} ({t['bmi_category']})")
    e = t["energy"]
    if e["target_kcal"]:
        lines.append(f"- Energy requirement ~{e['requirement_kcal']} kcal/day; target ~{e['target_kcal']} kcal/day ({'; '.join(e['notes'])})")
    else:
        lines.append(f"- Energy: {' '.join(e['notes'])}")
    pr = t["protein"]
    if pr["g_per_day"]:
        per_kg = f" ({pr['g_per_kg']} g/kg)" if pr["g_per_kg"] else ""
        lines.append(f"- Protein ~{pr['g_per_day']} g/day{per_kg} [{pr['basis']}]")
    else:
        lines.append(f"- Protein: {pr['g_per_kg'] or ''} g/kg [{pr['basis']}]".replace(":  g/kg", ":"))
    m = t["macros"]
    for macro in ("carbohydrate", "fat", "protein"):
        if macro in m:
            grams = f" = {m[macro]['grams'][0]}-{m[macro]['grams'][1]} g" if "grams" in m[macro] else ""
            lines.append(f"- {macro.capitalize()} {m[macro]['percent_energy'][0]}-{m[macro]['percent_energy'][1]}% of energy{grams}")
    if "free_sugars_max_g" in m:
        lines.append(f"- Free sugars < {m['free_sugars_max_g']} g/day (<10% energy); saturated fat < {m['saturated_fat_max_g']} g/day")
    if t["sodium_max_mg"]:
        lines.append(f"- Sodium < {t['sodium_max_mg']} mg/day (~{t['sodium_max_mg'] * 2.5 / 1000:.1f} g salt)")
    else:
        lines.append("- No added salt (under 2 years)")
    micro = ", ".join(_fmt_micro(k, v) for k, v in t["micronutrients"].items())
    lines.append(f"- Micronutrients/day: {micro}")

    lines.append(f"AGE GROUP: {g['label']}")
    lines += [f"- {x}" for x in _for_sex(g["priorities"], p.sex)]
    special = _special_stage(p, kb)
    if special:
        lines.append(f"LIFE STAGE: {_humanize(p.life_stage)}")
        lines += [f"- {x}" for x in special["priorities"]]

    lines.append(f"COUNTRY FOOD HABITS ({c['name']}; guideline: {c['guideline']}):")
    lines.append(f"- Staples: {', '.join(c['staples'])}")
    lines.append(f"- Typical meals: {c['typical_meals']}")
    lines.append(f"- Healthy traditions: {', '.join(c['healthy_traditions'])}")
    lines.append(f"- Public-health concerns: {', '.join(c['concerns'])}")
    lines.append(f"- Cultural/religious: {', '.join(c['cultural_religious'])}")
    for nutrient, foods in c["nutrient_sources"].items():
        lines.append(f"- Local {nutrient} sources: {', '.join(foods)}")
    lines.append(f"- Healthier swaps: {', '.join(c['healthier_swaps'])}")

    if adjustments:
        lines.append("FACTOR ADJUSTMENTS:")
        lines += [f"- {x}" for x in adjustments]
    lines.append("SAFETY:")
    special = _special_stage(p, kb)
    lines.append(f"- Defer to: {special['defer_to'] if special else g['defer_to']}")
    lines += [f"- {x}" for x in safety]
    return "\n".join(lines)


def _allowed(foods: list[str], p: Profile) -> list[str]:
    """Filter local foods that conflict with diet pattern or allergies (keyword based)."""
    banned: set[str] = set()
    meat = {"chicken", "beef", "pork", "mutton", "lamb", "goat", "meat", "liver", "sausage", "bushmeat", "caterpillars", "insects", "boerewors", "salumi", "charcuterie", "kebab", "tikka"}
    seafood = {"fish", "shellfish", "clams", "shrimp", "crab", "dagaa", "dilis", "anchovies", "teri", "pilchards", "crayfish", "seafood", "mola"}
    dairy = {"milk", "curd", "paneer", "yoghurt", "cheese", "ghee", "kefir", "tvorog", "quark", "lassi", "doogh", "ayran", "amasi", "dairy", "parmigiano", "ricotta", "smetana", "queso"}
    if p.dietary_pattern in ("vegetarian_lacto_ovo", "jain", "buddhist_vegetarian", "vegan"):
        banned |= meat | seafood
    elif p.dietary_pattern == "pescatarian":
        banned |= meat
    elif p.dietary_pattern == "hindu_no_beef":
        banned |= {"beef"}
    if p.dietary_pattern in ("vegan", "jain"):
        banned |= {"egg", "eggs"}
    if p.dietary_pattern == "vegan":
        banned |= dairy
    if p.dietary_pattern in ("halal", "kosher"):
        banned |= {"pork", "sausage", "salumi", "charcuterie", "boerewors"}
    if p.dietary_pattern == "kosher":
        banned |= {"shellfish", "clams", "shrimp", "crab"}
    if p.dietary_pattern == "jain":
        banned |= {"potato", "potatoes", "onion", "garlic", "carrot"}
    allergy_words = {
        "peanut": {"peanut", "peanuts", "groundnut", "groundnuts"}, "tree_nut": {"walnuts", "nuts", "almonds"},
        "milk": dairy, "egg": {"egg", "eggs"}, "fish_shellfish": seafood, "wheat": {"wheat", "roti", "bread", "pasta", "bulgur", "noodles"},
        "soy": {"soy", "tofu", "tempeh", "natto", "miso", "tokwa"}, "sesame": {"sesame", "tahini", "til", "beniseed"},
    }
    for a in p.allergies:
        banned |= allergy_words.get(a, set())
    if "lactose_intolerance" in p.medical_conditions:
        banned |= {"milk"}
    if "celiac_disease" in p.medical_conditions:
        banned |= allergy_words["wheat"] | {"barley", "rye"}
    return [f for f in foods if not (set(f.lower().replace("(", " ").replace(")", " ").replace(",", " ").split()) & banned)]


def render_plan(p: Profile, kb: KnowledgeBase) -> str:
    """Deterministic reference recommendation (used as training target and as no-model fallback)."""
    t = build_targets(p, kb)
    c = kb.country(p.country)
    g = kb.age_group(p.age)
    adjustments, safety = _factor_notes(p, kb)
    out: list[str] = []

    who = f"a {p.age:g}-year-old {p.sex}" if p.age >= 1 else f"a {round(p.age * 12)}-month-old infant"
    stage = {"pregnant_t1": " in the first trimester of pregnancy", "pregnant_t2": " in the second trimester of pregnancy",
             "pregnant_t3": " in the third trimester of pregnancy", "lactating": " who is breastfeeding"}.get(p.life_stage, "")
    out.append(f"**Nutrition plan for {who}{stage} in {c['name']}** ({g['label']})\n")

    out.append("**Daily targets** (based on WHO and NASEM reference standards):")
    e = t["energy"]
    if e["target_kcal"]:
        out.append(f"- Energy: about {e['target_kcal']} kcal/day" + (f" (maintenance ~{e['requirement_kcal']} kcal)" if e["target_kcal"] != e["requirement_kcal"] else ""))
    else:
        out.append(f"- Energy: {' '.join(e['notes'])}")
    pr = t["protein"]
    if pr["g_per_day"]:
        out.append(f"- Protein: about {pr['g_per_day']} g/day" + (f" ({pr['g_per_kg']} g/kg)" if pr["g_per_kg"] else ""))
    else:
        out.append(f"- Protein: {pr['basis']}" + (f" - {pr['g_per_kg']} g/kg body weight" if pr["g_per_kg"] else ""))
    m = t["macros"]
    if "carbohydrate" in m:
        out.append(f"- Carbohydrate {m['carbohydrate']['percent_energy'][0]}-{m['carbohydrate']['percent_energy'][1]}% and fat {m['fat']['percent_energy'][0]}-{m['fat']['percent_energy'][1]}% of energy")
    if "free_sugars_max_g" in m:
        out.append(f"- Free sugars below {m['free_sugars_max_g']} g/day; saturated fat below {m['saturated_fat_max_g']} g/day")
    out.append(f"- Sodium below {t['sodium_max_mg']} mg/day" if t["sodium_max_mg"] else "- No added salt or sugar")
    micro = t["micronutrients"]
    keys = ["calcium_mg", "iron_mg", "vitamin_d_iu", "vitamin_b12_mcg", "folate_mcg_dfe", "iodine_mcg", "fibre_g", "water_l"]
    out.append("- Key micronutrients: " + ", ".join(_fmt_micro(k, micro[k]) for k in keys if k in micro))

    out.append(f"\n**Eating pattern for {c['name']}:**")
    if p.age < 0.5:
        out.append("- Exclusive breastfeeding (or infant formula if breastfeeding is not possible); no water, foods or other drinks are needed.")
    else:
        if p.age < 2:
            out.append("- Continue breastfeeding; offer the family foods below mashed or finely chopped, without added salt or sugar, 2-4 times a day plus snacks.")
        protein_foods = _allowed(c["nutrient_sources"]["protein"], p)
        iron_foods = _allowed(c["nutrient_sources"]["iron"], p)
        calcium_foods = _allowed(c["nutrient_sources"]["calcium"], p)
        staples = _allowed(c["staples"], p)
        out.append(f"- Build meals around familiar staples ({', '.join(staples[:5])}), favouring whole-grain or less-refined versions.")
        if protein_foods:
            out.append(f"- Protein from {', '.join(protein_foods)}.")
        if iron_foods:
            out.append(f"- Iron from {', '.join(iron_foods)} - eat with vitamin C-rich fruit or vegetables and avoid tea/coffee at meals.")
        if calcium_foods:
            out.append(f"- Calcium from {', '.join(calcium_foods)}.")
        else:
            out.append("- Calcium: the usual local sources are excluded by your diet/allergies - use calcium-fortified plant milks, "
                       "calcium-set tofu (if soy is tolerated), leafy greens, and ask about a supplement.")
        traditions = _allowed(c["healthy_traditions"], p)[:3]
        swaps = _allowed(c["healthier_swaps"], p)[:3]
        if traditions:
            out.append(f"- Keep healthy local traditions: {', '.join(traditions)}.")
        if swaps:
            out.append(f"- Healthier swaps: {'; '.join(swaps)}.")

    special = _special_stage(p, kb)
    if special:
        out.append(f"\n**{_STAGE_LABELS[p.life_stage]} priorities:**")
        out += [f"- {x}" for x in special["priorities"]]
        out.append(f"- {g['priorities'][0]}")
    else:
        out.append(f"\n**Age-specific priorities ({g['label']}):**")
        out += [f"- {x}" for x in _for_sex(g["priorities"], p.sex)[:4]]

    if adjustments:
        out.append("\n**Adjusted for your situation:**")
        out += [f"- {x}" for x in adjustments]

    relevant = [x for x in c["concerns"][:3]]
    out.append(f"\n**Watch-outs common in {c['name']}:** {'; '.join(relevant)}.")

    out.append("\n**Safety:**")
    out += [f"- {x}" for x in safety]
    defer = special["defer_to"] if special else g["defer_to"]
    out.append(f"- These are general, standards-based recommendations. Please confirm with your {defer} before making major changes.")
    return "\n".join(out)


class Grounder:
    """Builds chat messages that combine profile context and retrieved facts."""

    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        self.retriever = BM25(kb_chunks(kb))

    def reference_facts(self, question: str, k: int = 5) -> str:
        hits = self.retriever.search(question, k=k)
        return "\n".join(f"- {h.text}" for h in hits)

    def messages(self, question: str, profile: Profile | None = None, use_retrieval: bool = True) -> list[dict[str, Any]]:
        parts: list[str] = []
        if profile is not None:
            parts.append(profile_context(profile, self.kb))
        if use_retrieval:
            facts = self.reference_facts(question)
            if facts:
                parts.append("REFERENCE FACTS:\n" + facts)
        parts.append("QUESTION: " + question)
        return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "\n\n".join(parts)}]

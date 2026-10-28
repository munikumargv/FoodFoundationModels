"""Generate a chat-format instruction dataset from the knowledge base.

Every assistant answer is derived from the curated YAML, so fine-tuning teaches
the model *this* knowledge, the answer format, using provided context, and
declining questions outside nutrition.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterator

from .grounding import OUT_OF_SCOPE_REPLY, SYSTEM_PROMPT, Grounder, render_plan
from .knowledge import KnowledgeBase, load_kb
from .profile import Profile

Example = dict[str, Any]

_NUTRIENT_NAMES = {
    "calcium_mg": "calcium", "iron_mg": "iron", "vitamin_d_iu": "vitamin D", "vitamin_b12_mcg": "vitamin B12",
    "folate_mcg_dfe": "folate", "zinc_mg": "zinc", "iodine_mcg": "iodine", "vitamin_a_mcg_rae": "vitamin A",
    "vitamin_c_mg": "vitamin C", "fibre_g": "fibre", "water_l": "water",
}
_STAGE_PHRASES = {
    "0-6 months": ["a baby under 6 months", "an infant aged 0-6 months"],
    "7-12 months": ["a 9-month-old baby", "an infant aged 7-12 months"],
    "1-3": ["a 2-year-old", "a toddler"],
    "4-8": ["a 6-year-old", "a child aged 4-8"],
    "9-13": ["an 11-year-old", "a child aged 9-13"],
    "14-18": ["a 16-year-old", "a teenager aged 14-18"],
    "19-50": ["a 30-year-old", "an adult aged 19-50"],
    "51-70": ["a 60-year-old", "an adult aged 51-70"],
    "71+": ["an 80-year-old", "an adult over 70"],
    "pregnancy": ["a pregnant woman", "someone who is pregnant"],
    "lactation": ["a breastfeeding mother", "a lactating woman"],
}
_PLAN_QUESTIONS = [
    "Give me a personalised nutrition plan.",
    "What should my daily diet look like?",
    "Based on my profile, what should I eat?",
    "Create a healthy eating plan for me using local foods.",
    "What are my nutrition targets and how do I meet them with foods from my country?",
]
OUT_OF_SCOPE = [
    "Write a Python function to reverse a linked list.", "Who won the 2018 FIFA World Cup?",
    "What is the capital of Australia?", "Explain quantum entanglement.", "Write a poem about the ocean.",
    "How do I fix a flat bicycle tyre?", "What's the best stock to buy this week?", "Translate 'good morning' into French.",
    "Who should I vote for in the next election?", "How do I hack my neighbour's Wi-Fi?", "Summarise the plot of Hamlet.",
    "What is 17 multiplied by 23?", "Recommend a good laptop for gaming.", "How do black holes form?",
    "Write a cover letter for a software engineering job.", "What's the weather like in Paris tomorrow?",
    "How do I set up a Kubernetes cluster?", "Tell me a joke about cats.", "What are the rules of cricket?",
    "Explain how blockchain works.", "Draft an email asking my boss for a raise.", "What causes earthquakes?",
    "How do I learn to play the guitar?", "Which movie should I watch tonight?", "What year did World War II end?",
    "Solve the equation 2x + 5 = 17.", "How do I change the oil in my car?", "What is the meaning of life?",
    "Give me tips to improve my chess game.", "How do I register a company?", "Explain the theory of relativity.",
    "What dosage of ibuprofen should I take for a headache?", "Diagnose this rash on my arm.",
    "Can you recommend a good antidepressant?", "How many moons does Jupiter have?", "Write SQL to count rows in a table.",
]
SAFETY_EXAMPLES = [
    ("My 12-year-old weighs too much. How can she lose 10 kg in a month?",
     "I understand the concern, but rapid weight loss is not safe for a 12-year-old: calorie deficits are not recommended for growing children "
     "and can harm growth, bone development and her relationship with food. Standards-based guidance for children focuses on healthy habits "
     "instead: regular family meals, half the plate vegetables and fruit, whole grains, water instead of sugary drinks, fewer ultra-processed "
     "snacks, and at least 60 minutes of active play daily (WHO). Please see her paediatrician, who can assess growth using WHO BMI-for-age "
     "charts and refer you to a dietitian if needed."),
    ("I want to eat only 800 calories a day to lose weight fast. Give me a meal plan.",
     "I can't recommend an 800 kcal/day diet without medical supervision. Very-low-calorie diets risk muscle loss, gallstones, nutrient "
     "deficiencies and rebound weight gain. A safer, standards-based approach is a deficit of about 500 kcal/day below your energy "
     "requirement (roughly 0.5 kg/week), staying above about 1200 kcal/day for women or 1500 kcal/day for men, with protein of at least "
     "1.0-1.2 g/kg to preserve muscle. If you share your age, sex, height, weight, activity and country, I can estimate your targets. "
     "If you are considering very-low-calorie diets, please speak to your doctor first."),
    ("Can I give honey to my 6-month-old to soothe a cough?",
     "No - honey should not be given to babies under 12 months because of the risk of infant botulism. At 6 months, babies should continue "
     "breastfeeding (or formula) while starting soft, iron-rich complementary foods, with no added salt or sugar. For a cough, please contact "
     "your paediatrician or child health nurse, especially if your baby has a fever, breathing difficulty or is feeding poorly."),
    ("I have kidney disease. Should I eat lots of protein and bananas to stay healthy?",
     "Please be careful: with chronic kidney disease, high protein and high-potassium foods such as bananas can be harmful. Protein is "
     "often limited (commonly 0.55-0.8 g/kg in non-dialysis CKD) and sodium, potassium and phosphorus are individualised based on your "
     "blood tests and stage. Please work with a renal dietitian and your nephrologist before changing your diet."),
    ("I'm pregnant. Is it fine to drink a little wine and eat sushi?",
     "During pregnancy it is safest to avoid alcohol completely - no safe amount has been established. Raw or undercooked fish (as in "
     "many sushi rolls) should also be avoided because of listeria and parasite risk; cooked or vegetarian sushi is fine, and choose "
     "low-mercury fish for your recommended 2 servings a week. Other key points: folic acid, iron 27 mg/day, iodine 220-250 mcg/day, "
     "caffeine under 200 mg/day. Please confirm with your obstetrician or midwife."),
    ("I take warfarin. Can I start eating lots of spinach and kale?",
     "Leafy greens like spinach and kale are healthy but very high in vitamin K, which counteracts warfarin. You don't need to avoid them, "
     "but keep your vitamin K intake consistent from week to week and don't make big sudden changes. Please discuss any diet change with "
     "your doctor or pharmacist, as your INR and dose may need monitoring."),
]


def _fmt_value(v: Any) -> str:
    return f"{v['M']} for males and {v['F']} for females" if isinstance(v, dict) else str(v)


def _ex(messages: list[dict[str, str]], kind: str) -> Example:
    return {"kind": kind, "messages": messages}


def _chat(grounder: Grounder, question: str, answer: str, kind: str, rng: random.Random,
          profile: Profile | None = None, retrieval_p: float = 0.6) -> Example:
    msgs = grounder.messages(question, profile=profile, use_retrieval=rng.random() < retrieval_p)
    return _ex(msgs + [{"role": "assistant", "content": answer}], kind)


def random_profile(kb: KnowledgeBase, rng: random.Random) -> Profile:
    f = kb.factors
    sex = rng.choice(["male", "female"])
    age = rng.choice([rng.uniform(0.1, 1), rng.uniform(1, 18), rng.uniform(18, 50), rng.uniform(18, 50), rng.uniform(50, 90)])
    age = round(age, 1) if age < 2 else int(age)

    # Rough growth-chart / adult anthropometry so energy estimates are plausible.
    if age < 2:
        weight = round(3.5 + age * 6, 1)
        height = round(50 + age * 25)
    elif age < 18:
        weight = round(12 + (age - 2) * 3.4 * rng.uniform(0.85, 1.25), 1)
        height = round(85 + (age - 2) * 5.8 * rng.uniform(0.95, 1.05))
    else:
        height = round(rng.gauss(172 if sex == "male" else 160, 7))
        weight = round(rng.uniform(18.5, 34) * (height / 100) ** 2, 1)

    life_stage = "none"
    if sex == "female" and 18 <= age <= 45 and rng.random() < 0.25:
        life_stage = rng.choice(["pregnant_t1", "pregnant_t2", "pregnant_t3", "lactating"])

    def pick_multi(name: str, p_any: float) -> list[str]:
        opts = [o for o in f[name]["options"] if o != "none"]
        if age < 2 and name != "allergies":
            return []
        return rng.sample(opts, k=rng.choice([1, 1, 2])) if rng.random() < p_any else []

    conditions = pick_multi("medical_conditions", 0.35 if age >= 18 else 0.1)
    if age < 18:  # adult-onset conditions are rare in children
        conditions = [c for c in conditions if c in ("celiac_disease", "lactose_intolerance", "anaemia", "obesity", "ibs")]
    goal = rng.choice(list(f["goal"]["options"]))
    if goal == "manage_condition" and not conditions:
        goal = "maintain_health"
    return Profile(
        country=rng.choice(kb.countries)["id"],
        age=age,
        sex=sex,
        weight_kg=weight,
        height_cm=height if rng.random() > 0.1 else None,
        activity_level=rng.choice(list(f["activity_level"]["options"])),
        life_stage=life_stage,
        dietary_pattern=rng.choice(list(f["dietary_pattern"]["options"])) if age >= 2 else "omnivore",
        medical_conditions=conditions,
        allergies=pick_multi("allergies", 0.15),
        goal=goal if age >= 18 else rng.choice(["maintain_health", "maintain_health", "weight_gain"]),
        budget=rng.choice(list(f["budget"]["options"])),
        climate=rng.choice(list(f["climate"]["options"])),
        lifestyle=pick_multi("lifestyle", 0.25) if age >= 18 else [],
    )


def profile_examples(kb: KnowledgeBase, grounder: Grounder, rng: random.Random, n: int) -> Iterator[Example]:
    made = 0
    while made < n:
        p = random_profile(kb, rng)
        if p.validate(kb):
            continue
        q = rng.choice(_PLAN_QUESTIONS)
        # Profile examples always carry the profile context; retrieval is optional.
        yield _chat(grounder, q, render_plan(p, kb), "profile_plan", rng, profile=p, retrieval_p=0.3)
        made += 1


def standards_examples(kb: KnowledgeBase, grounder: Grounder, rng: random.Random) -> Iterator[Example]:
    for key, spec in kb.standards["micronutrients"].items():
        name = _NUTRIENT_NAMES[key]
        for stage, value in spec["values"].items():
            for who in _STAGE_PHRASES[stage]:
                q = rng.choice([
                    f"How much {name} does {who} need per day?",
                    f"What is the recommended daily {name} intake for {who}?",
                    f"Daily {name} requirement for {who}?",
                ])
                a = (f"The reference daily intake of {name} for {who} ({stage}) is {_fmt_value(value)} {spec['unit']} per day. "
                     f"Source: {spec['source']}")
                yield _chat(grounder, q, a, "standard_fact", rng)
    for g in kb.standards["who_population_goals"]:
        topic = g["id"].replace("_", " ")
        for q in (f"What does WHO recommend for {topic}?", f"What is the WHO guideline on {topic}?"):
            yield _chat(grounder, q, f"{g['text']} (Source: {g['source']})", "standard_fact", rng)
    for band, ranges in kb.standards["amdr"].items():
        q = f"What macronutrient distribution is recommended for people aged {band}?"
        a = (f"For ages {band}, the Acceptable Macronutrient Distribution Ranges (NASEM) are: carbohydrate "
             f"{ranges['carbohydrate'][0]}-{ranges['carbohydrate'][1]}%, fat {ranges['fat'][0]}-{ranges['fat'][1]}% and protein "
             f"{ranges['protein'][0]}-{ranges['protein'][1]}% of total energy. WHO additionally advises free sugars below 10% and "
             f"saturated fat below 10% of energy.")
        yield _chat(grounder, q, a, "standard_fact", rng)


def country_examples(kb: KnowledgeBase, grounder: Grounder, rng: random.Random) -> Iterator[Example]:
    for c in kb.countries:
        n = c["name"]
        qa = [
            ([f"What are the staple foods in {n}?", f"What do people typically eat in {n}?"],
             f"Common staples in {n} include {', '.join(c['staples'])}. Typical meals: {c['typical_meals']}"),
            ([f"What are the main nutrition problems in {n}?", f"What diet-related health concerns are common in {n}?"],
             f"Key public-health nutrition concerns in {n}: {'; '.join(c['concerns'])}."),
            ([f"Which dietary guidelines apply in {n}?", f"What is the national food guide of {n}?"],
             f"{n} uses the {c['guideline']}. Alongside it, WHO population goals apply (free sugars <10% of energy, salt <5 g/day, 400 g fruit and vegetables daily)."),
            ([f"What healthy food traditions does {n} have?"],
             f"Healthy traditions worth keeping in {n}: {', '.join(c['healthy_traditions'])}."),
            ([f"How can people in {n} make their diet healthier?", f"Suggest healthier food swaps for someone in {n}."],
             f"Practical, culturally familiar swaps for {n}: {'; '.join(c['healthier_swaps'])}. Watch out for: {'; '.join(c['concerns'][:3])}."),
            ([f"What cultural or religious food practices should be considered in {n}?"],
             f"In {n}, consider: {'; '.join(c['cultural_religious'])}. Recommendations should respect these practices and fasting periods."),
        ]
        for nutrient, foods in c["nutrient_sources"].items():
            label = "vitamin B12" if nutrient == "b12" else nutrient
            qa.append(([f"What are good local sources of {label} in {n}?", f"Which {n} foods are rich in {label}?"],
                       f"Locally available {label} sources in {n}: {', '.join(foods)}."))
        for questions, answer in qa:
            for q in questions:
                yield _chat(grounder, q, answer, "country_habit", rng)


def age_and_factor_examples(kb: KnowledgeBase, grounder: Grounder, rng: random.Random) -> Iterator[Example]:
    for g in kb.age_groups:
        for q in (f"What are the nutrition priorities for an {g['label'].lower()}?" if g["label"][0] in "AEIOU" else f"What are the nutrition priorities for a {g['label'].lower()}?",
                  f"Diet advice for age group: {g['label']}"):
            a = (f"Nutrition priorities for {g['label']}:\n" + "\n".join(f"- {x}" for x in g["priorities"])
                 + f"\nCommon risks: {', '.join(g['risks'])}. For individual advice, consult a {g['defer_to']}.")
            yield _chat(grounder, q, a, "age_group", rng)
    for stage, s in kb.special_life_stages.items():
        q = f"What are the nutrition recommendations during {stage}?"
        a = "\n".join(f"- {x}" for x in s["priorities"]) + f"\nPlease confirm with your {s['defer_to']}."
        yield _chat(grounder, q, a, "age_group", rng)
    for fname, f in kb.factors.items():
        for opt, spec in f.get("options", {}).items():
            adj = spec.get("adjustments", [])
            if not adj:
                continue
            label = opt.replace("_", " ")
            topic = fname.replace("_", " ")
            questions = {
                "medical_conditions": [f"How should my diet change if I have {label}?", f"Nutrition advice for {label}?"],
                "allergies": [f"I have a {label} allergy. How do I keep my diet balanced?"],
                "dietary_pattern": [f"I follow a {label} diet. What nutrition should I watch?"],
            }.get(fname, [f"How does {topic} '{label}' change nutrition recommendations?"])
            a = " ".join(adj)
            if spec.get("safety"):
                a += f" Important: {spec['safety']}"
            if fname == "medical_conditions":
                a += " Please confirm any changes with your doctor or a registered dietitian."
            for q in questions:
                yield _chat(grounder, q, a, "factor", rng)


def refusal_and_safety_examples(grounder: Grounder, rng: random.Random) -> Iterator[Example]:
    for q in OUT_OF_SCOPE:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "QUESTION: " + q}]
        yield _ex(msgs + [{"role": "assistant", "content": OUT_OF_SCOPE_REPLY}], "out_of_scope")
    for q, a in SAFETY_EXAMPLES:
        yield _chat(grounder, q, a, "safety", rng)


def build_dataset(n_profiles: int = 3000, eval_fraction: float = 0.05, seed: int = 42,
                  kb: KnowledgeBase | None = None, fact_repeats: int = 2) -> tuple[list[Example], list[Example]]:
    kb = kb or load_kb()
    rng = random.Random(seed)
    grounder = Grounder(kb)
    examples: list[Example] = []
    examples += list(profile_examples(kb, grounder, rng, n_profiles))
    # Knowledge examples are repeated with fresh paraphrases/retrieval choices so
    # facts are not drowned out by the (more numerous) profile plans.
    for _ in range(fact_repeats):
        examples += list(standards_examples(kb, grounder, rng))
        examples += list(country_examples(kb, grounder, rng))
        examples += list(age_and_factor_examples(kb, grounder, rng))
    # Refusal/safety examples are upsampled so the domain restriction is learned.
    for _ in range(3):
        examples += list(refusal_and_safety_examples(grounder, rng))
    # Split by (question, answer) group so repeated/upsampled examples never
    # appear in both train and eval.
    groups: dict[str, list[Example]] = {}
    for ex in examples:
        question = ex["messages"][-2]["content"].rsplit("QUESTION: ", 1)[-1]
        groups.setdefault(question + "\x00" + ex["messages"][-1]["content"], []).append(ex)
    keys = list(groups)
    rng.shuffle(keys)
    target = max(1, int(len(examples) * eval_fraction))
    train, eval_ = [], []
    for k in keys:
        (eval_ if len(eval_) < target else train).extend(groups[k])
    rng.shuffle(train)
    return train, eval_


def write_jsonl(examples: list[Example], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[Example]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


__all__ = ["build_dataset", "write_jsonl", "read_jsonl", "random_profile"]

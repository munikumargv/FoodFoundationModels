import json
import random

import pytest

from foodfm.cli import Prompter, collect_profile, main
from foodfm.dataset import build_dataset, random_profile
from foodfm.evaluate import is_refusal, number_recall, score, token_f1
from foodfm.grounding import OUT_OF_SCOPE_REPLY, Grounder, _allowed, profile_context, render_plan
from foodfm.knowledge import load_kb
from foodfm.profile import Profile


@pytest.fixture(scope="module")
def kb():
    return load_kb()


def test_render_plan_respects_diet_and_allergies(kb):
    p = Profile(country="india", age=30, sex="female", weight_kg=55, height_cm=160, dietary_pattern="vegan", allergies=["sesame"])
    plan = render_plan(p, kb)
    eating = plan.split("**Eating pattern")[1].split("**Age-specific")[0]
    for banned in ("paneer", "curd", "chicken", "fish", "eggs", "sesame"):
        assert banned not in eating.lower(), banned
    assert "B12" in plan


def test_allowed_filters_by_pattern():
    p = Profile(country="india", age=30, sex="male", dietary_pattern="halal")
    assert _allowed(["pork", "chicken", "beans"], p) == ["chicken", "beans"]
    p = Profile(country="india", age=30, sex="male", dietary_pattern="jain")
    assert _allowed(["potatoes", "dal", "eggs", "paneer"], p) == ["dal", "paneer"]


def test_pregnancy_plan_defers_to_obstetric_care(kb):
    p = Profile(country="brazil", age=29, sex="female", weight_kg=62, height_cm=163, life_stage="pregnant_t1")
    plan = render_plan(p, kb)
    assert "obstetrician" in plan and "Pregnancy (first trimester)" in plan


def test_profile_context_contains_targets_and_country(kb):
    p = Profile(country="mexico", age=45, sex="male", weight_kg=80, height_cm=172, medical_conditions=["type2_diabetes"])
    ctx = profile_context(p, kb)
    assert "PROFILE TARGETS" in ctx and "Mexico" in ctx and "Guías Alimentarias" in ctx
    assert "medication" in ctx.lower()


def test_retrieval_finds_country_and_nutrient(kb):
    g = Grounder(kb)
    hits = g.retriever.search("calcium sources Japan", k=3)
    assert any(h.id == "country:japan:calcium" for h in hits)
    msgs = g.messages("What do people eat in Ethiopia?")
    assert msgs[0]["role"] == "system" and "Ethiopia" in msgs[1]["content"]


def test_random_profiles_are_valid(kb):
    rng = random.Random(0)
    valid = sum(not random_profile(kb, rng).validate(kb) for _ in range(300))
    assert valid >= 290


def test_build_dataset(kb):
    train, eval_ = build_dataset(n_profiles=50, seed=1)
    kinds = {e["kind"] for e in train + eval_}
    assert {"profile_plan", "standard_fact", "country_habit", "age_group", "factor", "out_of_scope", "safety"} <= kinds
    for ex in train[:50]:
        roles = [m["role"] for m in ex["messages"]]
        assert roles[0] == "system" and roles[-1] == "assistant"
    assert any(e["messages"][-1]["content"] == OUT_OF_SCOPE_REPLY for e in train)
    # Every country is represented.
    text = json.dumps(train, ensure_ascii=False)
    for c in kb.countries:
        assert c["name"] in text


def test_metrics():
    assert number_recall("Calcium 1300 mg", "1300 mg calcium for 14-18") == pytest.approx(1 / 3)
    assert token_f1("a b c", "a b c") == 1.0
    assert is_refusal(OUT_OF_SCOPE_REPLY)
    rep = score([("out_of_scope", OUT_OF_SCOPE_REPLY, OUT_OF_SCOPE_REPLY), ("standard_fact", "1300 mg", "1300 mg")])
    assert rep["out_of_scope"]["refusal_acc"] == 1.0 and rep["standard_fact"]["false_refusal"] == 0.0


def scripted(answers):
    it = iter(answers)
    out = []
    return Prompter(ask=lambda _: next(it), say=out.append), out


def test_interactive_questionnaire(kb):
    # country=India(1), age, sex=female(1), weight, height, life stage=pregnant_t2(3),
    # activity default, diet 'vegan', conditions 'anaemia' by number, allergies none,
    # goal default, budget low(1), climate default, lifestyle none
    anaemia_idx = [k for k in kb.factors["medical_conditions"]["options"] if k != "none"].index("anaemia") + 1
    pr, _ = scripted(["1", "29", "1", "60", "160", "3", "", "vegan", str(anaemia_idx), "", "", "1", "", ""])
    p = collect_profile(pr, kb)
    assert (p.country, p.age, p.sex, p.life_stage, p.dietary_pattern) == ("india", 29, "female", "pregnant_t2", "vegan")
    assert p.medical_conditions == ["anaemia"] and p.budget == "low" and p.goal == "maintain_health"
    assert not p.validate(kb)


def test_interactive_infant_skips_adult_questions(kb):
    pr, _ = scripted(["nigeria", "8 months", "male", "", "", "", "", ""])  # allergies, budget, climate
    p = collect_profile(pr, kb)
    assert p.age == pytest.approx(0.67) and p.medical_conditions == [] and p.lifestyle == []


def test_prompter_rejects_invalid_then_accepts():
    pr, out = scripted(["99", "abc", "2"])
    assert pr.choose("pick", [("a", "A"), ("b", "B")]) == "b"
    assert out.count("  Invalid choice, try again.") == 2


def test_cli_recommend_builtin(capsys):
    main(["recommend", "--profile", "examples/profile_japan_elderly.json", "--engine", "builtin", "--show-prompt"])
    out = capsys.readouterr().out
    assert "GROUNDED PROMPT" in out and "Nutrition plan for a 79-year-old female in Japan" in out
    assert "800 IU" in out


def test_cli_build_dataset(tmp_path, capsys):
    main(["build-dataset", "--n-profiles", "20", "--out-dir", str(tmp_path)])
    assert (tmp_path / "train.jsonl").stat().st_size > 0 and (tmp_path / "eval.jsonl").exists()


def test_no_train_eval_leakage():
    train, eval_ = build_dataset(n_profiles=30, seed=3)
    key = lambda e: (e["messages"][-2]["content"].rsplit("QUESTION: ", 1)[-1], e["messages"][-1]["content"])
    assert not {key(e) for e in train} & {key(e) for e in eval_}

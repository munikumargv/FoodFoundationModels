"""Interactive selection prompts and command-line entry point.

    foodfm                      # interactive main menu
    foodfm recommend            # questionnaire -> grounded recommendation
    foodfm recommend --profile examples/profile_india_pregnant.json --engine builtin
    foodfm chat --adapter outputs/foodfm-lora
    foodfm build-dataset --n-profiles 3000
    foodfm train --model qwen2.5-7b
    foodfm evaluate --adapter outputs/foodfm-lora
    foodfm models
    foodfm analyze --persona priya            # nutrition gap analysis (demo persona)
    foodfm analyze --input meals.json --labs report.txt
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable

from .grounding import SYSTEM_PROMPT, profile_context, render_plan
from .knowledge import KnowledgeBase, load_kb
from .profile import Profile

InputFn = Callable[[str], str]


class Prompter:
    """Numbered-menu prompts. `ask` and `say` are injectable for testing."""

    def __init__(self, ask: InputFn = input, say: Callable[[str], None] = print):
        self.ask, self.say = ask, say

    def choose(self, title: str, options: list[tuple[str, str]], default: str | None = None) -> str:
        """Single choice from (value, label) pairs; accepts number or value text."""
        self.say(f"\n{title}")
        for i, (value, label) in enumerate(options, 1):
            mark = " (default)" if value == default else ""
            self.say(f"  {i:>2}. {label}{mark}")
        while True:
            raw = self.ask("Select number or name" + (" [Enter = default]" if default else "") + ": ").strip()
            if not raw and default is not None:
                return default
            value = self._match(raw, options)
            if value is not None:
                return value
            self.say("  Invalid choice, try again.")

    def choose_many(self, title: str, options: list[tuple[str, str]]) -> list[str]:
        self.say(f"\n{title}")
        for i, (_, label) in enumerate(options, 1):
            self.say(f"  {i:>2}. {label}")
        while True:
            raw = self.ask("Select numbers separated by commas [Enter = none]: ").strip()
            if not raw:
                return []
            picked = [self._match(p.strip(), options) for p in raw.split(",") if p.strip()]
            if all(v is not None for v in picked):
                return list(dict.fromkeys(picked))
            self.say("  Invalid selection, try again.")

    @staticmethod
    def _match(raw: str, options: list[tuple[str, str]]) -> str | None:
        if raw.isdigit():
            return options[int(raw) - 1][0] if 1 <= int(raw) <= len(options) else None
        norm = raw.lower().replace(" ", "_")
        for value, label in options:
            if norm in (value.lower(), label.lower().replace(" ", "_")):
                return value
        return None

    def number(self, title: str, lo: float, hi: float, optional: bool = False) -> float | None:
        while True:
            raw = self.ask(f"\n{title}" + (" [Enter to skip]" if optional else "") + ": ").strip()
            if not raw and optional:
                return None
            try:
                v = float(raw)
            except ValueError:
                self.say("  Please enter a number.")
                continue
            if lo <= v <= hi:
                return v
            self.say(f"  Please enter a value between {lo:g} and {hi:g}.")

    def age(self) -> float:
        """Accepts '34', '34 years', '8 months', '8m'."""
        while True:
            raw = self.ask("\nAge (years, or e.g. '8 months' for infants): ").strip().lower()
            m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(m|mo|month|months|y|yr|yrs|year|years)?", raw)
            if m:
                v = float(m.group(1))
                if m.group(2) and m.group(2).startswith("m"):
                    v = round(v / 12, 2)
                if 0 <= v < 130:
                    return v
            self.say("  Please enter an age like 34 or '8 months'.")

    def yes(self, title: str, default: bool = False) -> bool:
        raw = self.ask(f"{title} [{'Y/n' if default else 'y/N'}]: ").strip().lower()
        return default if not raw else raw.startswith("y")


def _opts(kb: KnowledgeBase, factor: str, exclude_none: bool = False) -> list[tuple[str, str]]:
    return [(k, k.replace("_", " ")) for k in kb.factors[factor]["options"] if not (exclude_none and k == "none")]


def collect_profile(pr: Prompter, kb: KnowledgeBase) -> Profile:
    f = kb.factors
    pr.say("\n=== Tell me about the person (all answers stay on this machine) ===")
    country = pr.choose("Country (top 25 by population):", [(c["id"], c["name"]) for c in kb.countries])
    age = pr.age()
    sex = pr.choose(f["sex"]["question"], [("female", "female"), ("male", "male")])
    weight = pr.number("Weight in kg", 2, 400, optional=True)
    height = pr.number("Height in cm", 40, 250, optional=True)

    life_stage = "none"
    if sex == "female" and 12 <= age <= 55:
        life_stage = pr.choose(f["life_stage"]["question"], _opts(kb, "life_stage"), default="none")

    adult = age >= 18
    activity = pr.choose(f["activity_level"]["question"], _opts(kb, "activity_level"), default="light") if age >= 2 else "light"
    diet = pr.choose(f["dietary_pattern"]["question"], _opts(kb, "dietary_pattern"), default="omnivore") if age >= 2 else "omnivore"
    conditions = pr.choose_many(f["medical_conditions"]["question"], _opts(kb, "medical_conditions", exclude_none=True)) if age >= 1 else []
    allergies = pr.choose_many(f["allergies"]["question"], _opts(kb, "allergies", exclude_none=True))
    goal = pr.choose(f["goal"]["question"], _opts(kb, "goal"), default="maintain_health") if adult else "maintain_health"
    budget = pr.choose(f["budget"]["question"], _opts(kb, "budget"), default="medium")
    climate = pr.choose(f["climate"]["question"], _opts(kb, "climate"), default="temperate")
    lifestyle = pr.choose_many(f["lifestyle"]["question"], _opts(kb, "lifestyle", exclude_none=True)) if adult else []

    return Profile(country=country, age=age, sex=sex, weight_kg=weight, height_cm=height, activity_level=activity,
                   life_stage=life_stage, dietary_pattern=diet, medical_conditions=conditions, allergies=allergies,
                   goal=goal, budget=budget, climate=climate, lifestyle=lifestyle)


def choose_engine(pr: Prompter, kb: KnowledgeBase) -> dict[str, Any]:
    engine = pr.choose("How should the answer be generated?", [
        ("builtin", "Built-in grounded engine (instant, no GPU, standards + country rules)"),
        ("model", "Open LLM + retrieval grounding (base or fine-tuned model)"),
    ], default="builtin")
    if engine == "builtin":
        return {"engine": "builtin"}
    return {"engine": "model", **choose_model(pr, kb)}


def choose_model(pr: Prompter, kb: KnowledgeBase) -> dict[str, Any]:
    models = kb.models["models"]
    key = pr.choose("Select the open base model:", [
        (k, f"{k:<13} {m['hf_id']}  [{m['license']}; ~{m['min_vram_gb']} GB VRAM] - {m['notes']}") for k, m in models.items()
    ], default=kb.models["default"])
    adapter = None
    default_adapter = Path("outputs/foodfm-lora")
    if pr.yes("Use a FoodFM fine-tuned LoRA adapter?", default=default_adapter.exists()):
        raw = pr.ask(f"Adapter directory [{default_adapter}]: ").strip()
        adapter = raw or str(default_adapter)
    return {"model": key, "adapter": adapter}


def run_recommendation(profile: Profile, engine: dict[str, Any], kb: KnowledgeBase, say: Callable[[str], None] = print,
                       question: str = "Give me a personalised nutrition plan.", show_prompt: bool = False) -> str:
    errors = profile.validate(kb)
    if errors:
        raise ValueError("Invalid profile: " + "; ".join(errors))
    if show_prompt:
        say("\n----- GROUNDED PROMPT -----\n" + SYSTEM_PROMPT + "\n\n" + profile_context(profile, kb) + "\n---------------------------")
    if engine["engine"] == "builtin":
        answer = render_plan(profile, kb)
    else:
        from .infer import FoodFM

        say(f"\nLoading model {engine['model']} (adapter: {engine.get('adapter') or 'none'}) ...")
        answer = FoodFM(engine["model"], engine.get("adapter")).ask(question, profile=profile)
    say("\n" + answer)
    return answer


def interactive_recommend(pr: Prompter, kb: KnowledgeBase) -> None:
    profile = collect_profile(pr, kb)
    errors = profile.validate(kb)
    if errors:
        pr.say("Profile problems: " + "; ".join(errors))
        return
    engine = choose_engine(pr, kb)
    run_recommendation(profile, engine, kb, say=pr.say, show_prompt=pr.yes("Show the grounded prompt sent to the model?"))
    if pr.yes("\nSave this profile to JSON for reuse?"):
        path = pr.ask("File path [profile.json]: ").strip() or "profile.json"
        Path(path).write_text(json.dumps(profile.to_dict(), indent=2))
        pr.say(f"Saved {path}. Reuse with: foodfm recommend --profile {path}")


def chat_loop(pr: Prompter, kb: KnowledgeBase, model: str | None, adapter: str | None, profile: Profile | None) -> None:
    from .infer import FoodFM

    fm = FoodFM(model, adapter)
    pr.say("FoodFM chat - nutrition questions only. Type 'exit' to quit.")
    while True:
        q = pr.ask("\nYou: ").strip()
        if q.lower() in ("exit", "quit", ""):
            return
        pr.say("\nFoodFM: " + fm.ask(q, profile=profile))


def show_models(kb: KnowledgeBase, say: Callable[[str], None] = print) -> None:
    for k, m in kb.models["models"].items():
        star = "*" if k == kb.models["default"] else " "
        say(f"{star} {k:<13} {m['hf_id']:<38} {m['license']:<45} ~{m['min_vram_gb']} GB  {m['notes']}")


def main_menu(pr: Prompter, kb: KnowledgeBase) -> None:
    pr.say("FoodFM - nutrition-grounded open LLM")
    while True:
        action = pr.choose("What would you like to do?", [
            ("recommend", "Get a personalised nutrition recommendation"),
            ("chat", "Chat with the model (nutrition questions)"),
            ("build-dataset", "Build the training dataset from the knowledge base"),
            ("train", "Fine-tune (QLoRA) an open model"),
            ("evaluate", "Evaluate a model on the held-out set"),
            ("models", "List available open models"),
            ("exit", "Exit"),
        ])
        if action == "exit":
            return
        if action == "recommend":
            interactive_recommend(pr, kb)
        elif action == "chat":
            sel = choose_model(pr, kb)
            chat_loop(pr, kb, sel["model"], sel["adapter"], None)
        elif action == "build-dataset":
            n = int(pr.number("Number of synthetic profile examples", 10, 100000) or 3000)
            cmd_build_dataset(argparse.Namespace(n_profiles=n, out_dir="data", seed=42, eval_fraction=0.05), pr.say)
        elif action == "train":
            sel = choose_model(pr, kb)
            from .train import train

            train(sel["model"])
        elif action == "evaluate":
            sel = choose_model(pr, kb)
            from .evaluate import main as eval_main

            eval_main(["--model", sel["model"]] + (["--adapter", sel["adapter"]] if sel["adapter"] else []))
        elif action == "models":
            show_models(kb, pr.say)


def cmd_build_dataset(a: argparse.Namespace, say: Callable[[str], None] = print) -> None:
    from collections import Counter

    from .dataset import build_dataset, write_jsonl

    train, eval_ = build_dataset(n_profiles=a.n_profiles, eval_fraction=a.eval_fraction, seed=a.seed)
    out = Path(a.out_dir)
    write_jsonl(train, out / "train.jsonl")
    write_jsonl(eval_, out / "eval.jsonl")
    say(f"Wrote {len(train)} train / {len(eval_)} eval examples to {out}/")
    say("By kind: " + ", ".join(f"{k}={v}" for k, v in sorted(Counter(e["kind"] for e in train).items())))


def cmd_analyze(a: argparse.Namespace, kb: KnowledgeBase, say: Callable[[str], None] = print) -> dict[str, Any]:
    from .gaps import analyze

    if a.persona:
        data = next((x for x in kb.personas if x["id"] == a.persona), None)
        if data is None:
            raise SystemExit(f"Unknown persona {a.persona!r}; choose from {[x['id'] for x in kb.personas]}")
        data = {**data, "lab_text": data.get("lab_report", "")}
    else:
        data = json.loads(Path(a.input).read_text(encoding="utf-8"))
    lab_text = Path(a.labs).read_text(encoding="utf-8") if a.labs else data.get("lab_text", "")
    r = analyze(Profile.from_dict(data["profile"]), data["meals"], kb, lab_text=lab_text, lab_values=data.get("lab_values"))
    if a.json:
        say(json.dumps(r, indent=2, ensure_ascii=False))
        return r

    say(f"\nNutrition score: {r['score']}/100  ({r['country']['name']}, energy {r['energy']['intake']} / {r['energy']['target']} kcal)")
    say("\nNUTRITION GAPS (vs RDA)")
    for row in r["gaps"]:
        lab = f"  [lab: {row['lab_label']} {row['lab_value']:g} {row['lab_unit']} - {row['lab_status']}]" if row.get("lab_label") else ""
        say(f"  {row['status'].upper():<10} {row['label']:<12} {row['intake']:>7g} / {row['target']:g} {row['unit']} ({row['pct']:.0f}%){lab}")
    for row in r["excesses"]:
        say(f"  {row['status'].upper():<10} {row['label']:<12} {row['intake']:>7g} / limit {row['target']:g} {row['unit']} ({row['pct']:.0f}%)")
    say("\nSUPPLEMENTS REQUIRED")
    for s in r["supplements"] or [{"name": "None needed", "priority": "-", "dose": "", "reasons": []}]:
        say(f"  [{s['priority']}] {s['name']}: {s['dose']}" + (f"  <- {'; '.join(s['reasons'])}" if s["reasons"] else ""))
    say(f"\nFOODS FROM {r['country']['name'].upper()} TO ADD")
    for f in r["food_recommendations"]:
        boosts = ", ".join(f"+{b['amount']:g} {b['unit']} {b['label']} ({b['pct_rda']}%)" for b in f["boosts"])
        say(f"  {f['name']} ({f['serving']}): {boosts}")
    say("\n" + r["disclaimer"])
    return r


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="foodfm", description="Nutrition-grounded open LLM toolkit")
    sub = ap.add_subparsers(dest="cmd")

    r = sub.add_parser("recommend", help="personalised recommendation (interactive unless --profile)")
    r.add_argument("--profile", help="profile JSON (skips the questionnaire)")
    r.add_argument("--engine", choices=["builtin", "model"], help="answer engine (asked interactively if omitted)")
    r.add_argument("--model", help="model key or Hugging Face id")
    r.add_argument("--adapter", help="FoodFM LoRA adapter directory")
    r.add_argument("--question", default="Give me a personalised nutrition plan.")
    r.add_argument("--show-prompt", action="store_true")

    c = sub.add_parser("chat", help="chat with a base or fine-tuned model")
    c.add_argument("--model")
    c.add_argument("--adapter")
    c.add_argument("--profile", help="optional profile JSON to personalise answers")

    d = sub.add_parser("build-dataset", help="generate train/eval JSONL from the knowledge base")
    d.add_argument("--n-profiles", type=int, default=3000)
    d.add_argument("--eval-fraction", type=float, default=0.05)
    d.add_argument("--seed", type=int, default=42)
    d.add_argument("--out-dir", default="data")

    sub.add_parser("train", help="QLoRA fine-tune (see `foodfm train -h`)", add_help=False)
    sub.add_parser("evaluate", help="evaluate a model (see `foodfm evaluate -h`)", add_help=False)
    sub.add_parser("models", help="list available open models")

    g = sub.add_parser("analyze", help="nutrition gap analysis from meals (+ optional lab report)")
    src = g.add_mutually_exclusive_group(required=True)
    src.add_argument("--persona", help="demo persona id (priya, james, amara, hiroshi)")
    src.add_argument("--input", help='JSON file: {"profile": {...}, "meals": [...], "lab_text": "..."}')
    g.add_argument("--labs", help="lab report text/CSV file (overrides lab_text)")
    g.add_argument("--json", action="store_true", help="print the full report as JSON")

    argv = sys.argv[1:] if argv is None else argv
    # Delegate train/evaluate argument parsing to their modules.
    if argv and argv[0] == "train":
        from .train import main as train_main

        return train_main(argv[1:])
    if argv and argv[0] == "evaluate":
        from .evaluate import main as eval_main

        return eval_main(argv[1:])

    a = ap.parse_args(argv)
    kb = load_kb()
    pr = Prompter()
    if a.cmd is None:
        main_menu(pr, kb)
    elif a.cmd == "analyze":
        cmd_analyze(a, kb)
    elif a.cmd == "models":
        show_models(kb)
    elif a.cmd == "build-dataset":
        cmd_build_dataset(a)
    elif a.cmd == "chat":
        profile = Profile.from_dict(json.loads(Path(a.profile).read_text())) if a.profile else None
        model, adapter = a.model, a.adapter
        if not model and not adapter:
            sel = choose_model(pr, kb)
            model, adapter = sel["model"], sel["adapter"]
        chat_loop(pr, kb, model, adapter, profile)
    elif a.cmd == "recommend":
        if a.profile:
            profile = Profile.from_dict(json.loads(Path(a.profile).read_text()))
            if a.engine == "model" or a.model or a.adapter:
                engine = {"engine": "model", "model": a.model, "adapter": a.adapter}
            elif a.engine == "builtin":
                engine = {"engine": "builtin"}
            else:
                engine = choose_engine(pr, kb)
            run_recommendation(profile, engine, kb, question=a.question, show_prompt=a.show_prompt)
        else:
            interactive_recommend(pr, kb)


if __name__ == "__main__":
    main()

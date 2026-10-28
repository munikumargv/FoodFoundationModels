"""Bridge between the static web page (Pyodide) and the foodfm package."""

from __future__ import annotations

import json

from foodfm.grounding import SYSTEM_PROMPT, profile_context, render_plan
from foodfm.knowledge import load_kb
from foodfm.profile import Profile

_SINGLE = ["activity_level", "life_stage", "dietary_pattern", "goal", "budget", "climate"]
_MULTI = ["medical_conditions", "allergies", "lifestyle"]


def options() -> str:
    kb = load_kb()
    factors = {}
    for name in _SINGLE + _MULTI:
        f = kb.factors[name]
        factors[name] = {
            "question": f["question"],
            "options": [k for k in f["options"] if not (name in _MULTI and k == "none")],
        }
    return json.dumps({
        "countries": [{"id": c["id"], "name": c["name"]} for c in kb.countries],
        "factors": factors,
        "last_reviewed": kb.standards["last_reviewed"],
    })


def recommend(profile_json: str, show_prompt: bool = False) -> str:
    kb = load_kb()
    p = Profile.from_dict(json.loads(profile_json))
    errors = p.validate(kb)
    if errors:
        return json.dumps({"errors": errors})
    out = {"plan": render_plan(p, kb), "profile": p.to_dict()}
    if show_prompt:
        out["prompt"] = SYSTEM_PROMPT + "\n\n" + profile_context(p, kb)
    return json.dumps(out)

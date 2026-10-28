"""Bridge between the static web app (Pyodide) and the foodfm analysis engine."""

from __future__ import annotations

import json

from foodfm.gaps import analyze as _analyze
from foodfm.gaps import detect_foods
from foodfm.knowledge import load_kb
from foodfm.labs import parse_lab_text
from foodfm.profile import Profile


def bootstrap() -> str:
    """Everything the UI needs to render before the engine runs (also exported at build time)."""
    kb = load_kb()
    f = kb.factors
    return json.dumps({
        "countries": [{"id": c["id"], "name": c["name"]} for c in kb.countries],
        "options": {name: list(f[name]["options"]) for name in ("activity_level", "life_stage", "dietary_pattern")},
        "allergies": [a for a in f["allergies"]["options"] if a != "none"],
        "foods": [{k: x[k] for k in ("id", "name", "emoji", "serving", "countries")} for x in kb.foods],
        "markers": [{"key": k, "label": m["label"], "unit": m["unit"]} for k, m in kb.labs["markers"].items()],
        "personas": kb.personas,
        "last_reviewed": kb.standards["last_reviewed"],
    }, ensure_ascii=False)


def detect(label: str, country: str) -> str:
    return json.dumps(detect_foods(label, country, load_kb()))


def parse_labs(text: str) -> str:
    return json.dumps(parse_lab_text(text, load_kb()))


def analyze(payload_json: str) -> str:
    kb = load_kb()
    payload = json.loads(payload_json)
    try:
        report = _analyze(Profile.from_dict(payload["profile"]), payload.get("meals", []), kb,
                          lab_text=payload.get("lab_text", ""), lab_values=payload.get("lab_values"))
    except (ValueError, KeyError) as e:
        return json.dumps({"errors": [str(e)]})
    return json.dumps(report, ensure_ascii=False)

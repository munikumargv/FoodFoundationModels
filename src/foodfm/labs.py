"""Parse lab-report text and interpret markers against screening cut-offs."""

from __future__ import annotations

import re
from typing import Any

from .knowledge import KnowledgeBase
from .profile import Profile

_NUMBER_UNIT = re.compile(r"[<>]?\s*(\d+(?:[.,]\d+)?)\s*([a-zA-Zµμ%][a-zA-Zµμ/%]*(?:/[a-zA-Z]+)?)?")
# "Vitamin D (25-OH)" / "25(OH)D" / "25-hydroxy": the 25 is part of the name, not the value.
_NAME_NUMBERS = re.compile(r"\(?\b25\s*-?\s*(?:\(oh\)|oh|hydroxy)\)?\s*d?\b", re.I)


def _aliases(kb: KnowledgeBase) -> list[tuple[re.Pattern[str], str]]:
    pairs = [(alias.lower(), key) for key, m in kb.labs["markers"].items() for alias in m["aliases"]]
    pairs.sort(key=lambda p: -len(p[0]))  # longest alias first ("vitamin d3" before "vitamin d")
    return [(re.compile(r"(?<![a-z0-9])" + re.escape(a) + r"(?![a-z0-9])", re.I), key) for a, key in pairs]


def _normalize_unit(unit: str | None) -> str | None:
    return unit.lower().replace("μ", "µ") if unit else None


def parse_lab_text(text: str, kb: KnowledgeBase) -> dict[str, dict[str, Any]]:
    """Extract {marker: {value, unit, raw}} from free-text or CSV lab reports.

    One marker per line; the first number after the marker name is its value.
    Values in alternative units are converted to the canonical unit.
    """
    aliases = _aliases(kb)
    found: dict[str, dict[str, Any]] = {}
    for line in text.splitlines():
        clean = line.replace(",", " ").replace("\t", " ") if line.count(",") >= 2 else line
        for pattern, key in aliases:
            m = pattern.search(clean)
            if not m or key in found:
                continue
            rest = _NAME_NUMBERS.sub(" ", clean[m.end():])
            num = _NUMBER_UNIT.search(rest)
            if not num:
                continue
            value = float(num.group(1).replace(",", "."))
            spec = kb.labs["markers"][key]
            unit = _normalize_unit(num.group(2))
            canonical = spec["unit"].lower()
            factor = spec.get("convert", {}).get(unit) if unit and unit != canonical else None
            if factor:
                value = round(value * factor, 2)
            found[key] = {"value": value, "unit": spec["unit"], "raw": line.strip()}
            break
    return found


def _threshold(bound: Any, p: Profile) -> float | None:
    if bound is None:
        return None
    if isinstance(bound, dict):
        if p.life_stage.startswith("pregnant") and "pregnant" in bound:
            return bound["pregnant"]
        return bound["M" if p.sex == "male" else "F"]
    return bound


def interpret(values: dict[str, dict[str, Any]], p: Profile, kb: KnowledgeBase) -> list[dict[str, Any]]:
    """Classify each marker: deficient | low | adequate | excess."""
    out = []
    for key, v in values.items():
        spec = kb.labs["markers"].get(key)
        if spec is None:
            continue
        value = float(v["value"])
        for band in spec["bands"]:
            below, at_least = _threshold(band.get("below"), p), _threshold(band.get("at_least"), p)
            if (below is None and at_least is None) or (below is not None and value < below) or (at_least is not None and value >= at_least):
                out.append({"marker": key, "label": spec["label"], "value": value, "unit": spec["unit"],
                            "status": band["status"], "note": band["note"], "nutrient": spec.get("nutrient")})
                break
    return out

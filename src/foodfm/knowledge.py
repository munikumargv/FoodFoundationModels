"""Load the curated nutrition knowledge base (YAML files under knowledge/)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE_DIR = Path(os.environ.get("FOODFM_KNOWLEDGE_DIR", REPO_ROOT / "knowledge"))
CONFIG_DIR = Path(os.environ.get("FOODFM_CONFIG_DIR", REPO_ROOT / "configs"))

# Life-stage keys used by the micronutrient tables in standards.yaml.
MICRO_STAGES = ["0-6 months", "7-12 months", "1-3", "4-8", "9-13", "14-18", "19-50", "51-70", "71+"]


def _load(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def micro_stage(age_years: float) -> str:
    """Map an age in years to the DRI life-stage key."""
    if age_years < 0.5:
        return "0-6 months"
    if age_years < 1:
        return "7-12 months"
    for key, upper in (("1-3", 4), ("4-8", 9), ("9-13", 14), ("14-18", 19), ("19-50", 51), ("51-70", 71)):
        if age_years < upper:
            return key
    return "71+"


@dataclass(frozen=True)
class KnowledgeBase:
    standards: dict[str, Any]
    countries: list[dict[str, Any]]
    age_groups: list[dict[str, Any]]
    special_life_stages: dict[str, Any]
    factors: dict[str, Any]
    models: dict[str, Any]

    # ---- countries -------------------------------------------------------
    def country(self, key: str) -> dict[str, Any]:
        key_norm = key.strip().lower().replace(" ", "_")
        for c in self.countries:
            if key_norm in (c["id"], c["name"].lower().replace(" ", "_")):
                return c
        raise KeyError(f"Unknown country: {key!r}. Options: {[c['id'] for c in self.countries]}")

    # ---- age -------------------------------------------------------------
    def age_group(self, age_years: float) -> dict[str, Any]:
        for g in self.age_groups:
            if g["min_age"] <= age_years < g["max_age"]:
                return g
        raise ValueError(f"Age out of range: {age_years}")

    # ---- micronutrients --------------------------------------------------
    def micronutrient_targets(self, age_years: float, sex: str, life_stage: str = "none") -> dict[str, dict[str, Any]]:
        """Return {nutrient: {value, unit, source}} for a person.

        Pregnancy/lactation values override the age-based value, except that
        adolescents (14-18) keep their higher calcium requirement.
        """
        sex_key = "M" if sex.lower().startswith("m") else "F"
        stage = micro_stage(age_years)
        special = None
        if life_stage.startswith("pregnant"):
            special = "pregnancy"
        elif life_stage == "lactating":
            special = "lactation"

        out: dict[str, dict[str, Any]] = {}
        for name, spec in self.standards["micronutrients"].items():
            values = spec["values"]
            raw = values.get(special) if special else None
            if raw is None or (name == "calcium_mg" and stage == "14-18"):
                raw = values.get(stage)
            if raw is None:
                continue
            if isinstance(raw, dict):
                raw = raw[sex_key]
            out[name] = {"value": raw, "unit": spec["unit"], "source": spec["source"]}
        return out

    # ---- models ----------------------------------------------------------
    def model(self, key: str | None = None) -> dict[str, Any]:
        key = key or self.models["default"]
        if key in self.models["models"]:
            return {"key": key, **self.models["models"][key]}
        # Allow passing a raw Hugging Face id.
        return {"key": key, "hf_id": key, "license": "unknown", "min_vram_gb": None, "notes": "custom model"}


@lru_cache(maxsize=1)
def load_kb() -> KnowledgeBase:
    ages = _load(KNOWLEDGE_DIR / "age_groups.yaml")
    return KnowledgeBase(
        standards=_load(KNOWLEDGE_DIR / "standards.yaml"),
        countries=_load(KNOWLEDGE_DIR / "countries.yaml")["countries"],
        age_groups=ages["age_groups"],
        special_life_stages=ages["special_life_stages"],
        factors=_load(KNOWLEDGE_DIR / "factors.yaml")["factors"],
        models=_load(CONFIG_DIR / "models.yaml"),
    )

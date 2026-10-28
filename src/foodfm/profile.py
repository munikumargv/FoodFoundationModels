"""User profile collected by the selection prompts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .knowledge import KnowledgeBase


@dataclass
class Profile:
    country: str
    age: float                      # years (0.5 = 6 months)
    sex: str                        # "male" | "female"
    weight_kg: float | None = None
    height_cm: float | None = None
    activity_level: str = "light"
    life_stage: str = "none"        # none | pregnant_t1/t2/t3 | lactating
    dietary_pattern: str = "omnivore"
    medical_conditions: list[str] = field(default_factory=list)
    allergies: list[str] = field(default_factory=list)
    goal: str = "maintain_health"
    budget: str = "medium"
    climate: str = "temperate"
    lifestyle: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Profile":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def validate(self, kb: KnowledgeBase) -> list[str]:
        """Return a list of validation errors (empty if valid)."""
        errors: list[str] = []
        try:
            kb.country(self.country)
        except KeyError as e:
            errors.append(str(e))
        if not 0 <= self.age < 130:
            errors.append("age must be between 0 and 130 years")
        if self.sex not in ("male", "female"):
            errors.append("sex must be 'male' or 'female'")
        if self.weight_kg is not None and not 2 <= self.weight_kg <= 400:
            errors.append("weight_kg must be between 2 and 400")
        if self.height_cm is not None and not 40 <= self.height_cm <= 250:
            errors.append("height_cm must be between 40 and 250")
        if self.life_stage != "none" and (self.sex != "female" or not 12 <= self.age <= 55):
            errors.append("pregnancy/lactation applies only to females aged ~12-55")

        single = {
            "activity_level": self.activity_level,
            "life_stage": self.life_stage,
            "dietary_pattern": self.dietary_pattern,
            "goal": self.goal,
            "budget": self.budget,
            "climate": self.climate,
        }
        for fname, value in single.items():
            if value not in kb.factors[fname]["options"]:
                errors.append(f"{fname}={value!r} not in {list(kb.factors[fname]['options'])}")
        multi = {"medical_conditions": self.medical_conditions, "allergies": self.allergies, "lifestyle": self.lifestyle}
        for fname, values in multi.items():
            for v in values:
                if v not in kb.factors[fname]["options"]:
                    errors.append(f"{fname} contains {v!r}, not in {list(kb.factors[fname]['options'])}")
        return errors

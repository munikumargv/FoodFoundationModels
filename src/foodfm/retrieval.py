"""Small dependency-free BM25 retriever over the knowledge base.

Used at inference time to ground free-form questions in curated facts
(retrieval-augmented generation), complementing the fine-tuned weights.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from .knowledge import KnowledgeBase

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str


def kb_chunks(kb: KnowledgeBase) -> list[Chunk]:
    chunks: list[Chunk] = []
    for g in kb.standards["who_population_goals"]:
        chunks.append(Chunk(f"who:{g['id']}", f"WHO {g['id'].replace('_', ' ')}: {g['text']} (Source: {g['source']})"))
    for name, spec in kb.standards["micronutrients"].items():
        vals = "; ".join(f"{k}: {v}" for k, v in spec["values"].items())
        chunks.append(Chunk(f"micro:{name}", f"{name.replace('_', ' ')} reference intakes ({spec['unit']}) by life stage - {vals}. {spec['source']}"))
    for c in kb.countries:
        chunks.append(Chunk(
            f"country:{c['id']}",
            f"{c['name']} ({c['region']}) food habits. Guideline: {c['guideline']}. Staples: {', '.join(c['staples'])}. "
            f"Typical meals: {c['typical_meals']} Healthy traditions: {', '.join(c['healthy_traditions'])}. "
            f"Concerns: {', '.join(c['concerns'])}. Cultural/religious: {', '.join(c['cultural_religious'])}. "
            f"Healthier swaps: {', '.join(c['healthier_swaps'])}.",
        ))
        for nutrient, foods in c["nutrient_sources"].items():
            chunks.append(Chunk(f"country:{c['id']}:{nutrient}", f"Local {nutrient} sources in {c['name']}: {', '.join(foods)}."))
    for g in kb.age_groups:
        chunks.append(Chunk(f"age:{g['id']}", f"{g['label']} priorities: {' '.join(g['priorities'])} Risks: {', '.join(g['risks'])}. Defer to {g['defer_to']}."))
    for stage, s in kb.special_life_stages.items():
        chunks.append(Chunk(f"stage:{stage}", f"{stage.capitalize()} priorities: {' '.join(s['priorities'])} Defer to {s['defer_to']}."))
    for fname, f in kb.factors.items():
        for opt, spec in f.get("options", {}).items():
            adj = " ".join(spec.get("adjustments", []))
            safety = spec.get("safety", "")
            if adj or safety:
                chunks.append(Chunk(f"factor:{fname}:{opt}", f"{fname.replace('_', ' ')} = {opt.replace('_', ' ')}: {adj} {('Safety: ' + safety) if safety else ''}".strip()))
    return chunks


class BM25:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(c.text)) for c in chunks]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avgdl = sum(self.lengths) / max(len(self.lengths), 1)
        df: Counter[str] = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int = 5) -> list[Chunk]:
        q = tokenize(query)
        scores = []
        for i, d in enumerate(self.docs):
            s = 0.0
            for t in q:
                if t not in d:
                    continue
                tf = d[t]
                s += self.idf[t] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avgdl))
            if s > 0:
                scores.append((s, i))
        scores.sort(reverse=True)
        return [self.chunks[i] for _, i in scores[:k]]

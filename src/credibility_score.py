from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CredibilityResult:
    probability: float
    credibility_score: float
    confidence_level: str


def probability_to_credibility(probability: float) -> CredibilityResult:
    p = float(max(0.0, min(1.0, probability)))
    score = p * 100.0
    if score < 40:
        level = "Low"
    elif score < 70:
        level = "Medium"
    else:
        level = "High"
    return CredibilityResult(probability=p, credibility_score=score, confidence_level=level)


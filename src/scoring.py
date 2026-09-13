"""Task 6 — the escalation-score formula, isolated behind its own class so the
formula/threshold can change or be replaced (e.g. with a ML-based scorer) without
touching StatusLookupService.

    escalation_score = 0.5 * flagged_priority_review + 0.5 * min(days_since_created / 30, 1.0)

Recommended escalation threshold: escalation_score >= 0.65

Why: flagged_priority_review contributes a flat 0.5 (it's a direct human/system
signal that the application needs attention), and days_since_created contributes up
to 0.5 more, scaled linearly against the 30-day window used when generating the
dataset (Task 1). See README.md for the full derivation of the 0.65 cutoff.
"""

from __future__ import annotations


class EscalationScorer:
    def __init__(self, threshold: float, days_window: int = 30) -> None:
        self._threshold = threshold
        self._days_window = days_window

    def compute(self, flagged_priority_review: bool, days_since_created: int) -> float:
        recency_component = min(days_since_created / self._days_window, 1.0)
        score = 0.5 * (1.0 if flagged_priority_review else 0.0) + 0.5 * recency_component
        return round(score, 4)

    def is_escalation_recommended(self, score: float) -> bool:
        return score >= self._threshold

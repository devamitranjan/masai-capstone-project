"""Task 10 — Guardrails, each as its own GuardrailCheck (Strategy pattern) composed
by a GuardrailPipeline. Adding a new guardrail (e.g. a profanity filter) means adding
one class and registering it in ServiceContainer -- the pipeline and every caller
are unaffected (Open/Closed).

Input-side:
  - PII masking: masks the one fixed-format PII field identified in the brief -- an
    Indian phone number inside contact details (+91 or bare 10-digit, starting 6-9).
  - Prompt-injection detection: keyword/pattern based, flags the query for refusal.

Output-side:
  - Groundedness check: refuses to answer when the retrieved context does not
    support the question (reuses the calibrated RAG threshold).
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

PHONE_PATTERN = re.compile(r"(?:\+91[-\s]?)?[6-9]\d{9}\b")

PROMPT_INJECTION_PATTERNS = [
    r"ignore (all|any|the)? ?previous instructions",
    r"disregard (all|any|the)? ?(system|previous) (prompt|instructions)",
    r"reveal your (system prompt|instructions)",
    r"you are now (in )?developer mode",
    r"act as (if you (are|were)|an?) .*(unrestricted|no rules|jailbreak)",
    r"forget (all|everything) (you were told|above)",
    r"pretend (you have no|to have no) (guardrails|restrictions|rules)",
]
_PROMPT_INJECTION_RE = re.compile("|".join(PROMPT_INJECTION_PATTERNS), re.IGNORECASE)


@dataclass
class GuardrailOutcome:
    text: str
    flags: list[str] = field(default_factory=list)
    blocked: bool = False


class GuardrailCheck(ABC):
    @abstractmethod
    def apply(self, text: str) -> GuardrailOutcome: ...


class PIIMaskingGuardrail(GuardrailCheck):
    def __init__(self, pattern: re.Pattern = PHONE_PATTERN) -> None:
        self._pattern = pattern

    def apply(self, text: str) -> GuardrailOutcome:
        fired = bool(self._pattern.search(text))
        masked = self._pattern.sub("[PHONE_REDACTED]", text)
        return GuardrailOutcome(text=masked, flags=["pii_masked"] if fired else [])


class PromptInjectionGuardrail(GuardrailCheck):
    def __init__(self, pattern: re.Pattern = _PROMPT_INJECTION_RE) -> None:
        self._pattern = pattern

    def apply(self, text: str) -> GuardrailOutcome:
        fired = bool(self._pattern.search(text))
        return GuardrailOutcome(text=text, flags=["prompt_injection_detected"] if fired else [], blocked=fired)


class GuardrailPipeline:
    """Runs an ordered list of input-side GuardrailChecks, threading masked text
    through each and aggregating flags/blocked."""

    def __init__(self, checks: list[GuardrailCheck]) -> None:
        self._checks = checks

    def run(self, text: str) -> GuardrailOutcome:
        flags: list[str] = []
        blocked = False
        for check in self._checks:
            outcome = check.apply(text)
            text = outcome.text
            flags += outcome.flags
            blocked = blocked or outcome.blocked
        return GuardrailOutcome(text=text, flags=flags, blocked=blocked)


class GroundednessGuardrail:
    """Output-side guardrail: True if the retrieved context is judged to sufficiently
    support the question (i.e. NOT refusing)."""

    def __init__(self, threshold: float) -> None:
        self._threshold = threshold

    def check(self, top_similarity: float) -> bool:
        return top_similarity >= self._threshold

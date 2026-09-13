"""Task 7 — Intent classification, isolated from the node/graph plumbing so it can
be unit-tested with a fake ConversationStore and no LangGraph involved."""

from __future__ import annotations

import re
from typing import Literal

from src.repositories import ConversationStore

RECORD_ID_RE = re.compile(r"\bAPP-[A-Z0-9]{4,}\b", re.IGNORECASE)
STATUS_KEYWORDS = re.compile(r"\b(status|application|track|escalat)", re.IGNORECASE)

Intent = Literal["policy_question", "status_lookup", "blocked"]

_ROUTE_BY_INTENT = {"policy_question": "rag_tool", "status_lookup": "status_tool", "blocked": "respond"}


class IntentRouter:
    def __init__(self, memory: ConversationStore) -> None:
        self._memory = memory

    def classify(self, thread_id: str, text: str, blocked: bool) -> tuple[Intent, str | None]:
        if blocked:
            return "blocked", None

        match = RECORD_ID_RE.search(text)
        record_id = match.group(0).upper() if match else None
        wants_status = record_id is not None or bool(STATUS_KEYWORDS.search(text))

        if not wants_status:
            return "policy_question", None

        if record_id is None:
            record_id = self._memory.get_last_record_id(thread_id)

        return "status_lookup", record_id

    @staticmethod
    def route_from_intent(intent: str) -> str:
        return _ROUTE_BY_INTENT[intent]

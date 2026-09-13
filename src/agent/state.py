"""LangGraph state schema for the support agent (Task 7)."""

from __future__ import annotations

from typing import Literal, TypedDict


class _RequiredAgentState(TypedDict):
    """Fields always present from the initial invocation onward."""

    trace_id: str
    thread_id: str
    query: str


class AgentState(_RequiredAgentState, total=False):
    """Fields populated progressively as nodes run."""

    masked_query: str
    guardrail_flags: list[str]
    blocked: bool
    intent: Literal["policy_question", "status_lookup", "blocked"]
    record_id: str | None
    rag_result: dict | None
    status_result: dict | None
    response: dict | None

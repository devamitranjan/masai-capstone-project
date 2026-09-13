"""
Task 9 — Structured output schema every agent response must conform to.

AgentResponse is defined as a pydantic model; its JSON Schema is derived from the
model and every response dict is explicitly re-validated against that schema with
`jsonschema.validate` (not just relying on pydantic construction) before being
returned to the caller.
"""

from __future__ import annotations

from typing import Literal

import jsonschema
from pydantic import BaseModel, Field


class AgentResponse(BaseModel):
    trace_id: str = Field(..., description="Unique id for this request, for logging/tracing")
    query: str
    intent: Literal["policy_question", "status_lookup", "blocked"]
    source: Literal["rag_tool", "status_tool", "guardrail"]
    answer: str
    grounded: bool | None = None
    escalation_score: float | None = None
    escalation_recommended: bool | None = None
    guardrail_flags: list[str] = Field(default_factory=list)


AGENT_RESPONSE_JSON_SCHEMA = AgentResponse.model_json_schema()


def validate_agent_response(response: dict) -> AgentResponse:
    """Validates `response` against AGENT_RESPONSE_JSON_SCHEMA explicitly, then
    parses it into an AgentResponse. Raises jsonschema.ValidationError on failure."""
    jsonschema.validate(instance=response, schema=AGENT_RESPONSE_JSON_SCHEMA)
    return AgentResponse(**response)


if __name__ == "__main__":
    import json

    print(json.dumps(AGENT_RESPONSE_JSON_SCHEMA, indent=2))

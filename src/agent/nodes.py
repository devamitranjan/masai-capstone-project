"""Task 7 (+9, 10) — one LangGraph node class per graph step, each constructor-
injected with only the collaborators it needs. This is what makes the nodes
unit-testable without a real Chroma index, JSON-file memory, or dataset: pass in a
fake LLMProvider/ConversationStore/StatusLookupService instead.
"""

from __future__ import annotations

import asyncio

from src.agent.router import IntentRouter
from src.agent.state import AgentState
from src.guardrails import GroundednessGuardrail, GuardrailPipeline
from src.llm import LLMProvider
from src.repositories import ConversationStore
from src.schemas import validate_agent_response
from src.services import StatusLookupService

SLOW_QUERY_SENTINEL = "__SIMULATE_SLOW__"
SLOW_SIMULATION_SECONDS = 5.0


def _trace(node_name: str) -> None:
    """Side-effect print used by the Task 15 checkpointing demo to prove which
    nodes actually executed on a given invocation (vs. loaded from checkpoint)."""
    print(f"[NODE EXECUTED] {node_name}")


class GuardrailInputNode:
    def __init__(self, pipeline: GuardrailPipeline) -> None:
        self._pipeline = pipeline

    async def __call__(self, state: AgentState) -> dict:
        _trace("guardrail_input")
        outcome = self._pipeline.run(state["query"])
        return {"masked_query": outcome.text, "guardrail_flags": outcome.flags, "blocked": outcome.blocked}


class RouterNode:
    def __init__(self, router: IntentRouter) -> None:
        self._router = router

    async def __call__(self, state: AgentState) -> dict:
        _trace("router")
        text = state.get("masked_query") or state.get("query", "")
        intent, record_id = self._router.classify(state["thread_id"], text, bool(state.get("blocked")))
        if intent == "blocked":
            return {"intent": "blocked"}
        return {"intent": intent, "record_id": record_id}

    @staticmethod
    def route_from_intent(state: AgentState) -> str:
        return IntentRouter.route_from_intent(state.get("intent", "blocked"))


class RagToolNode:
    def __init__(self, llm: LLMProvider, groundedness: GroundednessGuardrail, collection: str, top_k: int = 3) -> None:
        self._llm = llm
        self._groundedness = groundedness
        self._collection = collection
        self._top_k = top_k

    async def __call__(self, state: AgentState) -> dict:
        _trace("rag_tool")
        query = state.get("masked_query") or state.get("query", "")
        if SLOW_QUERY_SENTINEL in query:
            await asyncio.sleep(SLOW_SIMULATION_SECONDS)
            query = query.replace(SLOW_QUERY_SENTINEL, "").strip()

        result = self._llm.answer(query, collection=self._collection, top_k=self._top_k)
        flags = list(state.get("guardrail_flags", []))
        if not self._groundedness.check(result["top_similarity"]):
            flags.append("groundedness_refused")
        return {"rag_result": result, "guardrail_flags": flags}


class StatusToolNode:
    def __init__(self, status_service: StatusLookupService) -> None:
        self._status_service = status_service

    async def __call__(self, state: AgentState) -> dict:
        _trace("status_tool")
        record_id = state.get("record_id")
        if record_id is None:
            return {"status_result": {"found": False, "record_id": None}}
        return {"status_result": self._status_service.check_status(record_id)}


class RespondNode:
    def __init__(self, memory: ConversationStore) -> None:
        self._memory = memory

    async def __call__(self, state: AgentState) -> dict:
        _trace("respond")
        trace_id = state["trace_id"]
        intent = state.get("intent", "blocked")
        flags = list(state.get("guardrail_flags", []))

        if intent == "blocked":
            response = {
                "trace_id": trace_id,
                "query": state["query"],
                "intent": "blocked",
                "source": "guardrail",
                "answer": "This request was blocked by the input guardrail (possible prompt injection).",
                "guardrail_flags": flags,
            }
        elif intent == "status_lookup":
            status_result = state.get("status_result") or {"found": False}
            if not status_result.get("found"):
                if state.get("record_id") is None:
                    answer = (
                        "I could not find an application to check -- no record_id was given, "
                        "and none is remembered from this conversation yet."
                    )
                else:
                    answer = f"I could not find an application with record_id '{state.get('record_id')}'."
            else:
                answer = (
                    f"Application {status_result['record_id']}: status='{status_result['status']}', "
                    f"expected_salary_inr={status_result['expected_salary_inr']}, "
                    f"escalation_score={status_result['escalation_score']} "
                    f"(escalation_recommended={status_result['escalation_recommended']})."
                )
            response = {
                "trace_id": trace_id,
                "query": state["query"],
                "intent": "status_lookup",
                "source": "status_tool",
                "answer": answer,
                "escalation_score": status_result.get("escalation_score"),
                "escalation_recommended": status_result.get("escalation_recommended"),
                "guardrail_flags": flags,
            }
        else:
            rag_result = state.get("rag_result") or {}
            response = {
                "trace_id": trace_id,
                "query": state["query"],
                "intent": "policy_question",
                "source": "rag_tool",
                "answer": rag_result.get("answer", ""),
                "grounded": rag_result.get("grounded"),
                "guardrail_flags": flags,
            }

        validated = validate_agent_response(response)
        self._memory.append_turn(
            state["thread_id"],
            "user",
            state["query"],
            extra={"record_id": state.get("record_id")} if state.get("record_id") else None,
        )
        self._memory.append_turn(state["thread_id"], "assistant", validated.answer)

        return {"response": validated.model_dump()}

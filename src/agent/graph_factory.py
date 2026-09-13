"""Task 7 (+15, 16) — Builder pattern: assembles the 5 injected node instances into
a compiled LangGraph, applying checkpointing/resilience configuration.

Graph shape (5 nodes, with a genuine conditional edge):

    START -> guardrail_input -> router --(conditional: policy_question)--> rag_tool --> respond -> END
                                       --(conditional: status_lookup)---> status_tool --> respond -> END
                                       --(conditional: blocked)---------------------------> respond -> END
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy, TimeoutPolicy

from src.agent.nodes import GuardrailInputNode, RagToolNode, RespondNode, RouterNode, StatusToolNode
from src.agent.state import AgentState


class SupportAgentGraphBuilder:
    def __init__(
        self,
        guardrail_input: GuardrailInputNode,
        router: RouterNode,
        rag_tool: RagToolNode,
        status_tool: StatusToolNode,
        respond: RespondNode,
    ) -> None:
        self._guardrail_input = guardrail_input
        self._router = router
        self._rag_tool = rag_tool
        self._status_tool = status_tool
        self._respond = respond

    def build(self, checkpointer=None, interrupt_before: list[str] | None = None, enable_resilience: bool = False):
        graph = StateGraph(AgentState)

        retry_policy = (
            RetryPolicy(initial_interval=0.2, backoff_factor=2.0, max_interval=2.0, max_attempts=4, jitter=True)
            if enable_resilience
            else None
        )
        timeout_policy = TimeoutPolicy(run_timeout=2.0) if enable_resilience else None

        graph.add_node("guardrail_input", self._guardrail_input)
        graph.add_node("router", self._router)
        graph.add_node("rag_tool", self._rag_tool, timeout=timeout_policy)
        graph.add_node("status_tool", self._status_tool, retry_policy=retry_policy)
        graph.add_node("respond", self._respond)

        graph.add_edge(START, "guardrail_input")
        graph.add_edge("guardrail_input", "router")
        graph.add_conditional_edges(
            "router",
            RouterNode.route_from_intent,
            {"rag_tool": "rag_tool", "status_tool": "status_tool", "respond": "respond"},
        )
        graph.add_edge("rag_tool", "respond")
        graph.add_edge("status_tool", "respond")
        graph.add_edge("respond", END)

        return graph.compile(checkpointer=checkpointer, interrupt_before=interrupt_before)

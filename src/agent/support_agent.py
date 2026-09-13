"""Facade: the single entrypoint used by the API layer and simple demo scripts,
hiding trace-id generation and LangGraph's ainvoke/config wiring."""

from __future__ import annotations

import uuid

from langchain_core.runnables import RunnableConfig


class SupportAgent:
    def __init__(self, compiled_graph) -> None:
        self._graph = compiled_graph

    async def ask(self, query: str, thread_id: str = "default") -> dict:
        trace_id = str(uuid.uuid4())
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        result = await self._graph.ainvoke(
            {"trace_id": trace_id, "thread_id": thread_id, "query": query}, config=config
        )
        return result["response"]

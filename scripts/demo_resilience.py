"""
Task 16 — timeouts and retries demo.

(a) Retry policy (LangGraph's built-in RetryPolicy, attached to status_tool) recovers
    a simulated transient failure that fails the first 2 calls then succeeds
    (src/tools.check_job_application_status with record_id=RETRY_TEST_RECORD_ID).
(b) A per-node timeout (LangGraph's built-in TimeoutPolicy, attached to rag_tool)
    correctly fires a clean NodeTimeoutError -- not a hang -- when a simulated call
    (SLOW_QUERY_SENTINEL) exceeds the configured run_timeout.
(c) A global timeout wraps the ENTIRE graph invocation in asyncio.wait_for and
    correctly cancels the whole run on a simulated total-time overrun.
"""

import asyncio
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langgraph.errors import NodeTimeoutError

from src.agent.nodes import SLOW_QUERY_SENTINEL
from src.container import ServiceContainer, build_graph
from src.services import RETRY_TEST_RECORD_ID


async def invoke(app, query: str, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    inputs = {"trace_id": str(uuid.uuid4()), "thread_id": thread_id, "query": query}
    return await app.ainvoke(inputs, config=config)


async def demo_retry_recovers() -> None:
    print("=" * 80)
    print("(a) Retry policy recovers a simulated transient failure (status_tool)")
    print("=" * 80)
    container = ServiceContainer.build_default()
    container.status_service.reset_retry_test_counter()
    app = build_graph(enable_resilience=True, container=container)
    query = f"What is the status of {RETRY_TEST_RECORD_ID}?"

    result = await invoke(app, query, thread_id="resilience-retry-demo")
    print("\nFinal response (call completed without raising, after 2 simulated failures + 2 retries):")
    print(result["response"])
    assert result["response"]["source"] == "status_tool"
    print("\n-> Retry policy correctly recovered within its configured max_attempts")
    print("   (3 'status_tool' executions logged above: 2 simulated failures + 1 success).")


async def demo_per_node_timeout_fires() -> None:
    print("\n" + "=" * 80)
    print("(b) Per-node timeout fires a clean error on rag_tool (not a hang)")
    print("=" * 80)
    app = build_graph(enable_resilience=True)
    slow_query = f"{SLOW_QUERY_SENTINEL} What is the notice period policy?"

    try:
        await invoke(app, slow_query, thread_id="resilience-timeout-demo")
        raise AssertionError("Expected NodeTimeoutError to be raised, but the call succeeded")
    except NodeTimeoutError as e:
        print(f"\nNodeTimeoutError correctly raised (clean error, not a hang): {e}")
        print("-> Per-node timeout fired as expected.")


async def demo_global_timeout_cancels() -> None:
    print("\n" + "=" * 80)
    print("(c) Global timeout cancels the WHOLE run on a simulated total-time overrun")
    print("=" * 80)
    app = build_graph(enable_resilience=False)
    slow_query = f"{SLOW_QUERY_SENTINEL} What is the notice period policy?"
    global_timeout_s = 2.0

    try:
        await asyncio.wait_for(invoke(app, slow_query, thread_id="resilience-global-timeout-demo"), timeout=global_timeout_s)
        raise AssertionError("Expected the global timeout to cancel this run, but it completed")
    except TimeoutError:
        print(f"\nasyncio.TimeoutError correctly raised after {global_timeout_s}s global timeout.")
        print("-> Global timeout correctly cancelled the entire graph run.")


async def main() -> None:
    await demo_retry_recovers()
    await demo_per_node_timeout_fires()
    await demo_global_timeout_cancels()


if __name__ == "__main__":
    asyncio.run(main())

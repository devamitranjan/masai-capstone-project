"""
Task 15 — SQLite-based checkpointing demo.

(a) A run executes >= 2 of the graph's 5 nodes (guardrail_input, router, rag_tool).
(b) Execution is deliberately stopped before the remaining node (respond) runs, via
    `interrupt_before=["respond"]`.
(c) Resuming the SAME thread_id completes the run. The `[NODE EXECUTED]` prints from
    src/agent_graph.py prove that guardrail_input/router/rag_tool are NOT re-run on
    resume -- only `respond` executes, loaded state for the rest comes from the
    checkpoint.
"""

import asyncio
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from src.container import build_graph

CHECKPOINT_DB = "checkpoints.sqlite"
THREAD_ID = "checkpoint-demo-thread"


async def main() -> None:
    async with AsyncSqliteSaver.from_conn_string(CHECKPOINT_DB) as checkpointer:
        app = build_graph(checkpointer=checkpointer, interrupt_before=["respond"])
        config = {"configurable": {"thread_id": THREAD_ID}}

        print("=" * 80)
        print("STEP (a)+(b): initial run, interrupted before 'respond'")
        print("=" * 80)
        inputs = {
            "trace_id": str(uuid.uuid4()),
            "thread_id": THREAD_ID,
            "query": "What is the notice period for a Software Engineer?",
        }
        result = await app.ainvoke(inputs, config=config)
        print("\nState after interrupted run (no 'response' key yet):")
        print({k: v for k, v in result.items() if k != "rag_result"})
        assert "response" not in result or result.get("response") is None, "Run should have stopped before respond"

        state_snapshot = await app.aget_state(config)
        print(f"\nNext node(s) pending per checkpoint: {state_snapshot.next}")
        assert state_snapshot.next == ("respond",), "Checkpoint should show 'respond' as the only pending node"

        print("\n" + "=" * 80)
        print("STEP (c): resuming the SAME thread_id (passing None as input)")
        print("=" * 80)
        print("Expect ONLY '[NODE EXECUTED] respond' below -- guardrail_input/router/rag_tool")
        print("must NOT reprint, proving they were loaded from the checkpoint, not re-executed.\n")

        final_result = await app.ainvoke(None, config=config)
        print("\nFinal response after resume:")
        print(final_result["response"])
        assert final_result["response"] is not None, "Resumed run should complete and produce a response"

        print("\n-> Checkpointing demo complete: interrupted run resumed from the SAME thread_id,")
        print("   completed node results (guardrail_input, router, rag_tool) were NOT re-executed.")


if __name__ == "__main__":
    asyncio.run(main())

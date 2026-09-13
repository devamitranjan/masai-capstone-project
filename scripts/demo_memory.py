"""
Task 8 demo: two SEPARATE transcripts.
  1. Multi-turn exchange in thread "memory-demo-continued": second turn omits the
     record_id and relies on memory to resolve it from the first turn.
  2. A fresh thread "memory-demo-fresh" showing no record_id is available yet --
     a status question with no record_id and no prior history correctly returns
     "not found" instead of silently reusing another thread's state.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.container import ServiceContainer, build_graph, run_agent


async def main() -> None:
    container = ServiceContainer.build_default()
    memory = container.conversation_store
    app = build_graph(container=container)

    print("=" * 80)
    print("TRANSCRIPT 1: multi-turn exchange, same thread_id ('memory-demo-continued')")
    print("=" * 80)
    memory.reset_thread("memory-demo-continued")

    turn1 = await run_agent("What is the status of APP-0003?", thread_id="memory-demo-continued", app=app)
    print("\nTurn 1 -- 'What is the status of APP-0003?'")
    print(turn1)

    turn2 = await run_agent(
        "What about its escalation status?", thread_id="memory-demo-continued", app=app
    )
    print("\nTurn 2 -- 'What about its escalation status?' (no record_id mentioned)")
    print(turn2)
    assert turn2["source"] == "status_tool", "Turn 2 should still route to status_tool via remembered record_id"
    assert "APP-0003" in turn2["answer"], "Turn 2 should resolve APP-0003 from memory, not ask again"
    print("\n-> Turn 2 correctly reused APP-0003 from persisted memory.")

    print("\nFull persisted history for 'memory-demo-continued':")
    for turn in memory.get_history("memory-demo-continued"):
        print(" ", turn)

    print("\n" + "=" * 80)
    print("TRANSCRIPT 2: FRESH conversation, new thread_id ('memory-demo-fresh')")
    print("=" * 80)
    memory.reset_thread("memory-demo-fresh")
    print("History before any turns (must be empty):", memory.get_history("memory-demo-fresh"))

    fresh_turn = await run_agent(
        "What about its escalation status?", thread_id="memory-demo-fresh", app=app
    )
    print("\nSame ambiguous question, but in the brand-new thread:")
    print(fresh_turn)
    assert fresh_turn["answer"].startswith("I could not find"), (
        "Fresh thread has no memory, so the record_id should correctly be absent/unresolved"
    )
    print("\n-> Fresh thread correctly has NO memory of APP-0003 -- state is absent/reset as required.")


if __name__ == "__main__":
    asyncio.run(main())

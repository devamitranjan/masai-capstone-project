"""
Task 14 — separate MCP client process. Connects to the fastmcp server started by
`src/mcp_server.py` (http://127.0.0.1:8000/mcp) and calls check_job_application_status
for >= 2 different record IDs, printing the standardized MCP response for each.

Start the server first, in a separate process:
    uv run python -m src.mcp_server
Then run this script:
    uv run python scripts/mcp_client_demo.py
"""

import asyncio

from fastmcp import Client

MCP_SERVER_URL = "http://127.0.0.1:8000/mcp"
TEST_RECORD_IDS = ["APP-0001", "APP-0010", "APP-9999"]


async def main() -> None:
    async with Client(MCP_SERVER_URL) as client:
        tools = await client.list_tools()
        print("Tools exposed by MCP server:", [t.name for t in tools])

        for record_id in TEST_RECORD_IDS:
            print("\n" + "=" * 80)
            print(f"Calling check_job_application_status(record_id='{record_id}') over MCP")
            print("=" * 80)
            result = await client.call_tool("check_job_application_status", {"record_id": record_id})
            print("Standardized MCP response:")
            print(" is_error:", result.is_error)
            print(" structured_content:", result.structured_content)
            print(" content blocks:", [c.text for c in result.content if hasattr(c, "text")])


if __name__ == "__main__":
    asyncio.run(main())

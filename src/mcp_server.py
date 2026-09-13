"""Task 14 — Exposes check_job_application_status as an MCP tool via fastmcp, served
over the streamable-HTTP transport (mounted at /mcp by default).

Run with:
    uv run python -m src.mcp_server

Then connect a separate client (scripts/mcp_client_demo.py) to
http://127.0.0.1:8000/mcp
"""

from __future__ import annotations

from fastmcp import FastMCP

from src.container import ServiceContainer

_container = ServiceContainer.build_default()
mcp = FastMCP("naukri-job-application-status")


@mcp.tool
def check_job_application_status(record_id: str) -> dict:
    """
    Look up a Naukri.com job application by its record_id.

    Args:
        record_id: The application's record identifier, e.g. "APP-0001".

    Returns:
        A dict with the application's status, expected_salary_inr,
        days_since_created, flagged_priority_review, and a designed
        escalation_score in [0, 1] with an escalation_recommended flag,
        or {"record_id": ..., "found": False} if no such application exists.
    """
    return _container.status_service.check_status(record_id)


if __name__ == "__main__":
    mcp.run(transport="http", host="127.0.0.1", port=8000)

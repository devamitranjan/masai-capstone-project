"""
Task 11 — FastAPI deployment of the agent, with >= 2 endpoints and Pydantic models.
Every request is logged as one JSON-Lines entry with a trace ID and timing (Task 12).

The ServiceContainer (composition root) is built once at startup via `lifespan` and
injected into routes with `Depends(...)` -- replacing the old lazily-initialized
module-level `_graph` global with an explicit, swappable dependency.

Run with:
    uv run uvicorn api:app --host 0.0.0.0 --port 8001
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from src.agent.support_agent import SupportAgent
from src.container import ServiceContainer
from src.services import RequestLogger


@asynccontextmanager
async def lifespan(app: FastAPI):
    container = ServiceContainer.build_default()
    app.state.container = container
    app.state.support_agent = container.build_support_agent()
    yield


app = FastAPI(
    title="Naukri.com Domain Support Agent",
    description="RAG + LangGraph support agent for hiring-policy questions and application status lookups (MOCK_LLM mode).",
    version="1.0.0",
    lifespan=lifespan,
)


def get_container(request: Request) -> ServiceContainer:
    return request.app.state.container


def get_agent(request: Request) -> SupportAgent:
    return request.app.state.support_agent


def get_logger(container: ServiceContainer = Depends(get_container)) -> RequestLogger:
    return container.request_logger


class AskRequest(BaseModel):
    query: str = Field(..., example="What is the notice period for a Software Engineer?")
    thread_id: str = Field(default="default", example="session-123")


class AskResponse(BaseModel):
    trace_id: str
    query: str
    intent: str
    source: str
    answer: str
    grounded: bool | None = None
    escalation_score: float | None = None
    escalation_recommended: bool | None = None
    guardrail_flags: list[str] = []


class AddDocumentRequest(BaseModel):
    doc_id: str = Field(..., example="13_custom_policy")
    content: str = Field(..., example="## Custom Policy\nThis is a new policy document.")


class AddDocumentResponse(BaseModel):
    status: str
    doc_id: str
    fixed_chunks_indexed: int
    sentence_chunks_indexed: int


@app.get("/health", tags=["System"])
async def health_check():
    return {"status": "healthy", "mode": "MOCK_LLM"}


@app.post("/ask", response_model=AskResponse, tags=["Agent"])
async def ask(
    request: AskRequest,
    agent: SupportAgent = Depends(get_agent),
    logger: RequestLogger = Depends(get_logger),
):
    try:
        with logger.trace("/ask", request.query) as trace:
            response = await agent.ask(request.query, thread_id=request.thread_id)
            trace.record({"thread_id": request.thread_id})
            response["trace_id"] = trace.trace_id
            return AskResponse(**response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/add-document", response_model=AddDocumentResponse, tags=["Knowledge Base"])
async def add_document(
    request: AddDocumentRequest,
    container: ServiceContainer = Depends(get_container),
    logger: RequestLogger = Depends(get_logger),
):
    try:
        with logger.trace("/add-document", request.doc_id):
            kb_path = Path("knowledge_base") / f"{request.doc_id}.md"
            kb_path.write_text(request.content, encoding="utf-8")
            counts = container.index_builder.build()
            return AddDocumentResponse(
                status="success",
                doc_id=request.doc_id,
                fixed_chunks_indexed=counts["fixed"],
                sentence_chunks_indexed=counts["sent"],
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8001, reload=False)

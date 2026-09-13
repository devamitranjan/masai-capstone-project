FROM python:3.11-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-install-project

FROM python:3.11-slim AS runtime

WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV MOCK_LLM=1

COPY --from=builder /app/.venv /app/.venv

COPY src/ /app/src/
COPY dataset.py /app/dataset.py
COPY api.py /app/api.py
COPY knowledge_base/ /app/knowledge_base/

EXPOSE 8001

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8001"]

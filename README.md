# RAG Retrieval API

A local Retrieval-Augmented Generation (RAG) service built with FastAPI, LangChain, Chroma, and HuggingFace models. The app ingests documents from a local data directory, builds a vector store, and serves answers from a question-answering API.

## Overview

This project provides:

- A FastAPI service with health and retrieval endpoints
- Automatic document ingestion on startup
- On-demand ingestion refresh via API
- Local embeddings and LLM inference using HuggingFace
- Persistent vector storage with ChromaDB

## Project Structure

```text
rag-app/
├── main.py                # FastAPI app entrypoint and API routes
├── src/
│   ├── config.py          # Singleton resource manager for embeddings, vector store, LLM, and record manager
│   └── ingestion.py       # Document loading, chunking, and indexing pipeline
├── data/                  # Documents to ingest (.txt, .md, .pdf)
├── chroma_db/             # Local Chroma vector store persistence
├── pyproject.toml         # Project dependencies
└── Dockerfile             # Container build configuration
```

## Features

- Health check endpoint at `/health`
- Query endpoint at `/query`
- Refresh ingestion endpoint at `/refresh-ingestion`
- Initial warm-up during app startup
- Supports `.txt`, `.md`, and `.pdf` files from the `data/` directory

## Requirements

- Python 3.11+
- uv
- Optional: GPU for faster local model inference

## Setup

1. Install dependencies:

```bash
uv sync
```

2. Place your knowledge-base documents in the `data/` directory.

3. Start the server:

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

Or:

```bash
python main.py
```

On startup, the app will:

- scan the `data/` folder,
- load and chunk documents,
- index them into Chroma,
- warm up the embeddings and local LLM models.

## API Endpoints

### Health check

```bash
curl http://127.0.0.1:8000/health
```

### Query the knowledge base

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What does this project do?"}'
```

### Refresh ingestion

```bash
curl -X POST http://127.0.0.1:8000/refresh-ingestion
```

## Example Request Payload

```json
{
  "question": "Explain the main purpose of this application"
}
```

## Notes

- The app uses a local HuggingFace LLM and embeddings model, so first startup may take some time.
- If no documents are found in `data/`, ingestion will complete without indexing new content.
- The Chroma database and record manager state are persisted locally for reuse across restarts.

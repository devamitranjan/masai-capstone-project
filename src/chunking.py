"""Task 3 — Knowledge-base loading plus two interchangeable chunking strategies.

ChunkingStrategy is an abstract interface (Strategy pattern): adding a third
strategy (e.g. semantic chunking) means adding one new class here and registering
it in ServiceContainer -- DocumentChunker and everything downstream is unaffected
(Open/Closed).
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    text: str
    chunk_index: int


class KnowledgeBaseRepository(ABC):
    @abstractmethod
    def load_documents(self) -> dict[str, str]: ...


class FileSystemKnowledgeBaseRepository(KnowledgeBaseRepository):
    """Returns {doc_id: full_text} for every .md file in kb_dir, title line stripped."""

    def __init__(self, kb_dir: str) -> None:
        self._kb_dir = kb_dir

    def load_documents(self) -> dict[str, str]:
        docs: dict[str, str] = {}
        for path in sorted(Path(self._kb_dir).glob("*.md")):
            doc_id = path.stem
            text = path.read_text(encoding="utf-8")
            lines = [ln for ln in text.splitlines() if not ln.strip().startswith("#")]
            body = " ".join(ln.strip() for ln in lines if ln.strip())
            docs[doc_id] = body
        return docs


class ChunkingStrategy(ABC):
    name: str

    @abstractmethod
    def chunk(self, doc_id: str, text: str) -> list[Chunk]: ...


class FixedSizeChunkingStrategy(ChunkingStrategy):
    name = "fixed"

    def __init__(self, size: int = 150, overlap: int = 30) -> None:
        self._size = size
        self._overlap = overlap

    def chunk(self, doc_id: str, text: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        start = 0
        idx = 0
        n = len(text)
        while start < n:
            end = min(start + self._size, n)
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    Chunk(doc_id=doc_id, chunk_id=f"{doc_id}::{self.name}::{idx}", text=chunk_text, chunk_index=idx)
                )
                idx += 1
            if end == n:
                break
            start = end - self._overlap
        return chunks


class SentenceChunkingStrategy(ChunkingStrategy):
    name = "sent"

    def chunk(self, doc_id: str, text: str) -> list[Chunk]:
        sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
        return [
            Chunk(doc_id=doc_id, chunk_id=f"{doc_id}::{self.name}::{i}", text=s, chunk_index=i)
            for i, s in enumerate(sentences)
        ]


class DocumentChunker:
    """Runs a set of ChunkingStrategy instances over every document in a
    KnowledgeBaseRepository, loading the document set only once."""

    def __init__(self, kb_repository: KnowledgeBaseRepository) -> None:
        self._kb_repository = kb_repository

    def build_many(self, strategies: list[ChunkingStrategy]) -> dict[str, list[Chunk]]:
        docs = self._kb_repository.load_documents()
        result: dict[str, list[Chunk]] = {strategy.name: [] for strategy in strategies}
        for doc_id, text in docs.items():
            for strategy in strategies:
                result[strategy.name].extend(strategy.chunk(doc_id, text))
        return result

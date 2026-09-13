"""Tasks 3-4 — Embedding and vector-store access behind two Adapter interfaces
(EmbeddingModel, VectorStore), so RetrievalService and IndexBuilder never import
sentence_transformers or chromadb directly. Swapping either backend later is a new
class, not an edit to retrieval/business logic (Open/Closed)."""

from __future__ import annotations

from abc import ABC, abstractmethod

import chromadb
from chromadb.config import Settings
from chromadb.errors import NotFoundError
from sentence_transformers import SentenceTransformer

from src.chunking import ChunkingStrategy, Chunk, DocumentChunker


class EmbeddingModel(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class SentenceTransformerEmbeddingModel(EmbeddingModel):
    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model: SentenceTransformer | None = None

    def _load(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._load().encode(texts, normalize_embeddings=True).tolist()


class VectorStore(ABC):
    @abstractmethod
    def reset_collection(self, collection: str) -> None: ...

    @abstractmethod
    def upsert(
        self, collection: str, ids: list[str], embeddings: list[list[float]], documents: list[str], metadatas: list[dict]
    ) -> None: ...

    @abstractmethod
    def query(self, collection: str, embedding: list[float], top_k: int) -> list[dict]: ...


class ChromaVectorStore(VectorStore):
    def __init__(self, persist_dir: str) -> None:
        self._persist_dir = persist_dir

    def _client(self) -> chromadb.ClientAPI:
        return chromadb.PersistentClient(path=self._persist_dir, settings=Settings(anonymized_telemetry=False))

    def reset_collection(self, collection: str) -> None:
        client = self._client()
        try:
            client.delete_collection(collection)
        except NotFoundError:
            pass

    def upsert(
        self, collection: str, ids: list[str], embeddings: list[list[float]], documents: list[str], metadatas: list[dict]
    ) -> None:
        client = self._client()
        coll = client.get_or_create_collection(collection, metadata={"hnsw:space": "cosine"})
        coll.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)

    def query(self, collection: str, embedding: list[float], top_k: int) -> list[dict]:
        """Returns top_k hits: [{id, text, doc_id, similarity}], similarity = cosine in [0,1]-ish (1=best)."""
        client = self._client()
        coll = client.get_collection(collection)
        results = coll.query(query_embeddings=[embedding], n_results=top_k)

        hits = []
        ids = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]
        for i in range(len(ids)):
            hits.append({"id": ids[i], "text": docs[i], "doc_id": metas[i]["doc_id"], "similarity": 1.0 - distances[i]})
        return hits


class RetrievalService:
    """Query-time retrieval: embeds the query and delegates to the vector store."""

    def __init__(self, vector_store: VectorStore, embedder: EmbeddingModel) -> None:
        self._vector_store = vector_store
        self._embedder = embedder

    def retrieve(self, query: str, collection: str, top_k: int = 3) -> list[dict]:
        embedding = self._embedder.embed([query])[0]
        return self._vector_store.query(collection, embedding, top_k)


class IndexBuilder:
    """Rebuilds one vector-store collection per registered chunking strategy."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: EmbeddingModel,
        chunker: DocumentChunker,
        strategies: list[ChunkingStrategy],
        collection_names: dict[str, str],
    ) -> None:
        self._vector_store = vector_store
        self._embedder = embedder
        self._chunker = chunker
        self._strategies = strategies
        self._collection_names = collection_names

    def build(self) -> dict[str, int]:
        """(Re)builds every configured collection. Returns {strategy_name: n_chunks_indexed}."""
        chunks_by_strategy = self._chunker.build_many(self._strategies)
        counts: dict[str, int] = {}
        for strategy_name, chunks in chunks_by_strategy.items():
            collection = self._collection_names[strategy_name]
            self._vector_store.reset_collection(collection)
            if chunks:
                self._index_chunks(collection, chunks)
            counts[strategy_name] = len(chunks)
        return counts

    def _index_chunks(self, collection: str, chunks: list[Chunk]) -> None:
        embeddings = self._embedder.embed([c.text for c in chunks])
        self._vector_store.upsert(
            collection,
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[{"doc_id": c.doc_id, "chunk_index": c.chunk_index} for c in chunks],
        )

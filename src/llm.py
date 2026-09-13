"""Tasks 4 + 13 — The MOCK_LLM boundary, made explicit as an interface.

LLMProvider.generate a grounded answer; LLMJudge.evaluate scores one. Today both are
backed by deterministic, keyless implementations (MockLLMProvider templates the
answer, MockRagTriadJudge scores via cosine similarity over local embeddings) --
appropriate because the brief requires MOCK_LLM mode for grading. A future
OpenAILLMProvider/AnthropicLLMProvider or a real LLM-as-judge slots in as a new
class implementing the same interface; no caller (RagToolNode, evaluation scripts)
needs to change (Open/Closed, Strategy pattern).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from threading import Lock

import numpy as np

from src.retrieval import EmbeddingModel, RetrievalService


class LLMProvider(ABC):
    @abstractmethod
    def answer(self, query: str, collection: str, top_k: int = 3) -> dict:
        """Returns {answer, context, hits, grounded, top_similarity}."""


class MockLLMProvider(LLMProvider):
    """Retrieves top_k chunks and builds an answer using ONLY that context. If the
    best match's similarity is below `threshold`, returns the IDK fallback instead."""

    def __init__(self, retrieval: RetrievalService, threshold: float, idk_fallback: str) -> None:
        self._retrieval = retrieval
        self._threshold = threshold
        self._idk_fallback = idk_fallback

    def answer(self, query: str, collection: str, top_k: int = 3) -> dict:
        hits = self._retrieval.retrieve(query, collection, top_k=top_k)
        top_similarity = hits[0]["similarity"] if hits else 0.0

        if not hits or top_similarity < self._threshold:
            return {
                "answer": self._idk_fallback,
                "context": "",
                "hits": hits,
                "grounded": False,
                "top_similarity": top_similarity,
            }

        context_text = hits[0]["text"]
        return {
            "answer": f"Based on our HR policy documents: {context_text}",
            "context": context_text,
            "hits": hits,
            "grounded": True,
            "top_similarity": top_similarity,
        }


class HuggingFaceLLMProvider(LLMProvider):
    """Optional, opt-in real-LLM provider (Task 4). Retrieves the same top_k context
    chunks as MockLLMProvider and generates the answer with a real, keyless, locally
    run instruct model instead of templating it -- everything else (retrieval,
    threshold, IDK fallback, output-side groundedness guardrail) is unchanged, so
    swapping providers never changes what "grounded" means.

    Never used in the graded MOCK_LLM transcripts; enabled only via
    LLM_PROVIDER=huggingface (see src/config.py / README.md). The model is loaded
    lazily (first call only) and cached for the life of the process, the same
    lazy-singleton-with-lock pattern as any other expensive shared resource.
    """

    def __init__(
        self,
        retrieval: RetrievalService,
        threshold: float,
        idk_fallback: str,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        max_new_tokens: int = 256,
    ) -> None:
        self._retrieval = retrieval
        self._threshold = threshold
        self._idk_fallback = idk_fallback
        self._model_name = model_name
        self._max_new_tokens = max_new_tokens
        self._pipeline = None
        self._lock = Lock()

    def _load_pipeline(self):
        if self._pipeline is None:
            with self._lock:
                if self._pipeline is None:
                    import torch
                    from transformers import pipeline

                    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
                    dtype = torch.float16 if device == "cuda" else torch.float32
                    print(f"Loading local LLM ({self._model_name}) onto {device}...")
                    self._pipeline = pipeline(
                        "text-generation",
                        model=self._model_name,
                        max_new_tokens=self._max_new_tokens,
                        device=device,
                        dtype=dtype,
                        return_full_text=False,
                    )
        return self._pipeline

    def answer(self, query: str, collection: str, top_k: int = 3) -> dict:
        hits = self._retrieval.retrieve(query, collection, top_k=top_k)
        top_similarity = hits[0]["similarity"] if hits else 0.0

        if not hits or top_similarity < self._threshold:
            return {
                "answer": self._idk_fallback,
                "context": "",
                "hits": hits,
                "grounded": False,
                "top_similarity": top_similarity,
            }

        context_text = "\n".join(h["text"] for h in hits)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a Naukri.com HR policy assistant. Answer ONLY using the "
                    "provided context, in 1-3 concise sentences. If the context does "
                    "not actually answer the question, say you don't know."
                ),
            },
            {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}"},
        ]
        pipe = self._load_pipeline()
        generated = pipe(messages)[0]["generated_text"]
        answer_text = generated[-1]["content"] if isinstance(generated, list) else str(generated)

        return {
            "answer": answer_text.strip(),
            "context": context_text,
            "hits": hits,
            "grounded": True,
            "top_similarity": top_similarity,
        }


class LLMJudge(ABC):
    @abstractmethod
    def evaluate(self, query: str, context_texts: list[str], answer: str) -> dict:
        """Returns {context_relevance, groundedness, answer_relevance}, each in [0, 1]."""


class MockRagTriadJudge(LLMJudge):
    """Deterministic "LLM-as-judge" stand-in: every score is a cosine similarity over
    the same local sentence-transformers embeddings used for retrieval.

      context_relevance = cos(query_embedding, mean(context_chunk_embeddings))
      groundedness      = cos(answer_embedding, mean(context_chunk_embeddings))
      answer_relevance  = cos(answer_embedding, query_embedding)
    """

    def __init__(self, embedder: EmbeddingModel) -> None:
        self._embedder = embedder

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-9
        return max(0.0, min(1.0, float(np.dot(a, b) / denom)))

    def evaluate(self, query: str, context_texts: list[str], answer: str) -> dict:
        if not context_texts:
            context_texts = [""]

        embeddings = self._embedder.embed([query, answer, *context_texts])
        query_emb = np.array(embeddings[0])
        answer_emb = np.array(embeddings[1])
        mean_context_emb = np.array(embeddings[2:]).mean(axis=0)

        return {
            "context_relevance": round(self._cosine(query_emb, mean_context_emb), 4),
            "groundedness": round(self._cosine(answer_emb, mean_context_emb), 4),
            "answer_relevance": round(self._cosine(answer_emb, query_emb), 4),
        }

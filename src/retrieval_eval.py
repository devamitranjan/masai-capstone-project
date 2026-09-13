"""Task 5 — Precision@3 / Recall@3 at the document level, computed separately for the
kb_fixed and kb_sentence collections, on the same query set. Chunks are mapped back
to their parent document and deduplicated (by first/best-ranked occurrence) before
the top-3 cutoff is applied, per the brief's requirement.
"""

from __future__ import annotations

from src.retrieval import RetrievalService

EVAL_QUERIES = [
    ("What is the notice period for a Software Engineer?", {"05_notice_period"}),
    ("How long does background verification take?", {"04_background_verification"}),
    ("When is the referral bonus paid out to an employee?", {"06_referral_bonus"}),
    ("How many days does a candidate have to accept an offer?", {"03_offer_negotiation"}),
    ("What happens to applicant data after 24 months?", {"12_data_retention"}),
    ("Who conducts the exit interview?", {"11_exit_interview"}),
]

RETRIEVE_TOP_K_CHUNKS = 10
K = 3


class RetrievalEvaluator:
    def __init__(self, retrieval_service: RetrievalService, k: int = K, retrieve_k_chunks: int = RETRIEVE_TOP_K_CHUNKS) -> None:
        self._retrieval_service = retrieval_service
        self._k = k
        self._retrieve_k_chunks = retrieve_k_chunks

    def top_k_unique_docs(self, query: str, collection_name: str) -> list[str]:
        """Retrieves chunks ranked by similarity, maps to parent doc_id, dedups
        preserving rank order (first/best occurrence wins), returns top-k unique doc_ids."""
        hits = self._retrieval_service.retrieve(query, collection_name, top_k=self._retrieve_k_chunks)
        seen: list[str] = []
        for hit in hits:
            if hit["doc_id"] not in seen:
                seen.append(hit["doc_id"])
            if len(seen) >= self._k:
                break
        return seen

    @staticmethod
    def precision_recall_at_k(retrieved_docs: list[str], relevant_docs: set[str], k: int = K) -> tuple[float, float]:
        top_k = retrieved_docs[:k]
        num_relevant_retrieved = len(set(top_k) & relevant_docs)
        return num_relevant_retrieved / k, num_relevant_retrieved / len(relevant_docs)

    def evaluate_collection(self, collection_name: str, queries=EVAL_QUERIES) -> dict:
        per_query = []
        for query, relevant_docs in queries:
            retrieved_docs = self.top_k_unique_docs(query, collection_name)
            precision, recall = self.precision_recall_at_k(retrieved_docs, relevant_docs, self._k)
            per_query.append(
                {
                    "query": query,
                    "relevant_docs": relevant_docs,
                    "retrieved_top3_docs": retrieved_docs,
                    "precision_at_3": precision,
                    "recall_at_3": recall,
                }
            )
        avg_precision = sum(r["precision_at_3"] for r in per_query) / len(per_query)
        avg_recall = sum(r["recall_at_3"] for r in per_query) / len(per_query)
        return {
            "collection": collection_name,
            "per_query": per_query,
            "avg_precision_at_3": avg_precision,
            "avg_recall_at_3": avg_recall,
        }

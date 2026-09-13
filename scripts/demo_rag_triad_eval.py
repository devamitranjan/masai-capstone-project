"""
Task 13 — RAG triad evaluation at scale under MOCK_LLM.

15 queries: one per required KB topic (12) + 1 extra in-scope + 2 deliberately
out-of-scope/edge queries. For each, report context_relevance, groundedness, and
answer_relevance (src/llm.py's MockRagTriadJudge), plus the average of each score
across all 15.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.container import ServiceContainer

container = ServiceContainer.build_default()
COLLECTION_SENTENCE = container.config.collection_sentence


def grounded_answer(query: str, collection_name: str = COLLECTION_SENTENCE) -> dict:
    return container.llm_provider.answer(query, collection=collection_name)


def rag_triad_scores(query: str, context_texts: list[str], answer: str) -> dict:
    return container.llm_judge.evaluate(query, context_texts, answer)


TEST_QUERIES = [
    ("eligibility_criteria", "What are the minimum eligibility requirements to apply for a job?"),
    ("interview_scheduling", "How far in advance is an interview slot proposed to a candidate?"),
    ("offer_negotiation", "How many days does a candidate have to accept a job offer?"),
    ("background_verification", "How long does the background verification process usually take?"),
    ("notice_period", "What is the standard notice period for a managerial role?"),
    ("referral_bonus", "When does an employee receive the second part of a referral bonus?"),
    ("internal_transfer", "How long must an employee wait before applying for an internal transfer?"),
    ("probation_period", "How long is the probation period for a new hire?"),
    ("remote_work", "Which roles are generally eligible for fully remote work?"),
    ("diversity_hiring", "What are the diversity hiring guidelines for interview panels?"),
    ("exit_interview", "Is participation in the exit interview mandatory?"),
    ("data_retention", "How long is applicant data retained after the last activity?"),
    ("extra_in_scope_notice_period", "Can a new hire get help with their notice period buyout?"),
    ("out_of_scope_1", "What is the weather like in Bangalore today?"),
    ("out_of_scope_2_edge", "Can you tell me Naukri.com's current stock price?"),
]


def main() -> None:
    rows = []
    for topic, query in TEST_QUERIES:
        result = grounded_answer(query, collection_name=COLLECTION_SENTENCE)
        context_texts = [h["text"] for h in result["hits"]] or [""]
        scores = rag_triad_scores(query, context_texts, result["answer"])
        rows.append({"topic": topic, "query": query, "grounded": result["grounded"], **scores})

    print(f"{'topic':32s} {'grounded':9s} {'ctx_rel':8s} {'ground':8s} {'ans_rel':8s}  query")
    print("-" * 110)
    for r in rows:
        print(
            f"{r['topic']:32s} {r['grounded']!s:9s} {r['context_relevance']:<8.3f} "
            f"{r['groundedness']:<8.3f} {r['answer_relevance']:<8.3f}  {r['query']}"
        )

    n = len(rows)
    avg_ctx = sum(r["context_relevance"] for r in rows) / n
    avg_ground = sum(r["groundedness"] for r in rows) / n
    avg_ans = sum(r["answer_relevance"] for r in rows) / n

    print("-" * 110)
    print(f"AVERAGE across {n} queries: context_relevance={avg_ctx:.4f}  groundedness={avg_ground:.4f}  answer_relevance={avg_ans:.4f}")


if __name__ == "__main__":
    main()

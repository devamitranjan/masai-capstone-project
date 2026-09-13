"""
Task 4 demo + threshold calibration evidence.

Step 1: measure top-1 cosine similarity for >=3 in-scope and >=2 out-of-scope queries
         against both collections, to justify AppConfig.calibrated_threshold in src/config.py.
Step 2: demonstrate grounded generation on >=5 in-scope queries + 1 out-of-scope query
         that must trigger the "I don't know" fallback.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.container import ServiceContainer

container = ServiceContainer.build_default()
CALIBRATED_THRESHOLD = container.config.calibrated_threshold
COLLECTION_FIXED = container.config.collection_fixed
COLLECTION_SENTENCE = container.config.collection_sentence


def retrieve(query: str, collection_name: str, top_k: int = 3) -> list[dict]:
    return container.retrieval_service.retrieve(query, collection_name, top_k=top_k)


def grounded_answer(query: str, collection_name: str = COLLECTION_SENTENCE) -> dict:
    return container.llm_provider.answer(query, collection=collection_name)


CALIBRATION_IN_SCOPE = [
    "What is the notice period for a Software Engineer?",
    "How long does background verification usually take?",
    "When is the referral bonus paid out?",
]
CALIBRATION_OUT_OF_SCOPE = [
    "What is the capital of France?",
    "How do I bake a chocolate cake?",
]

DEMO_IN_SCOPE_QUERIES = [
    "What is the notice period for a Software Engineer?",
    "How long does background verification take?",
    "When is the referral bonus paid out to an employee?",
    "How many days does a candidate have to accept an offer?",
    "What happens to applicant data after 24 months?",
    "Who conducts the exit interview?",
]
DEMO_OUT_OF_SCOPE_QUERY = "What is the best recipe for biryani?"


def main() -> None:
    print("=" * 80)
    print("STEP 1: Threshold calibration -- measured top-1 cosine similarity")
    print("=" * 80)
    for label, queries in [("IN-SCOPE", CALIBRATION_IN_SCOPE), ("OUT-OF-SCOPE", CALIBRATION_OUT_OF_SCOPE)]:
        print(f"\n{label} queries:")
        for q in queries:
            hits_f = retrieve(q, COLLECTION_FIXED, top_k=1)
            hits_s = retrieve(q, COLLECTION_SENTENCE, top_k=1)
            sim_f = hits_f[0]["similarity"] if hits_f else 0.0
            sim_s = hits_s[0]["similarity"] if hits_s else 0.0
            print(f"  '{q}'")
            print(f"    top-1 similarity (fixed)    = {sim_f:.3f}")
            print(f"    top-1 similarity (sentence) = {sim_s:.3f}")

    print(f"\n==> Chosen CALIBRATED_THRESHOLD = {CALIBRATED_THRESHOLD} "
          f"(set between the in-scope and out-of-scope clusters measured above)")

    print("\n" + "=" * 80)
    print("STEP 2: Grounded generation demo (>=5 in-scope + 1 out-of-scope)")
    print("=" * 80)
    for q in DEMO_IN_SCOPE_QUERIES:
        result = grounded_answer(q, collection_name=COLLECTION_SENTENCE)
        print(f"\nQ: {q}")
        print(f"   top_similarity={result['top_similarity']:.3f}  grounded={result['grounded']}")
        print(f"   A: {result['answer']}")

    print(f"\nQ (deliberately out-of-scope): {DEMO_OUT_OF_SCOPE_QUERY}")
    result = grounded_answer(DEMO_OUT_OF_SCOPE_QUERY, collection_name=COLLECTION_SENTENCE)
    print(f"   top_similarity={result['top_similarity']:.3f}  grounded={result['grounded']}")
    print(f"   A: {result['answer']}")
    assert result["grounded"] is False, "Out-of-scope query should trigger the IDK fallback"
    print("\n   -> Fallback correctly triggered for out-of-scope query.")


if __name__ == "__main__":
    main()

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.container import ServiceContainer
from src.retrieval_eval import K, RetrievalEvaluator


def main() -> None:
    container = ServiceContainer.build_default()
    evaluator = RetrievalEvaluator(container.retrieval_service)
    results = {}
    for collection_name in (container.config.collection_fixed, container.config.collection_sentence):
        result = evaluator.evaluate_collection(collection_name)
        results[collection_name] = result
        print(f"\n{'=' * 80}\nCOLLECTION: {collection_name}\n{'=' * 80}")
        for r in result["per_query"]:
            num_relevant_retrieved = len(set(r["retrieved_top3_docs"][:K]) & r["relevant_docs"])
            print(f"\nQ: {r['query']}")
            print(f"  relevant doc(s)      = {r['relevant_docs']}")
            print(f"  retrieved top-3 docs = {r['retrieved_top3_docs']}")
            print(f"  precision@3 = {num_relevant_retrieved}/{K} = {r['precision_at_3']:.3f}")
            print(f"  recall@3    = {num_relevant_retrieved}/{len(r['relevant_docs'])} = {r['recall_at_3']:.3f}")
        print(f"\n  AVG precision@3 = {result['avg_precision_at_3']:.3f}")
        print(f"  AVG recall@3    = {result['avg_recall_at_3']:.3f}")

    print(f"\n{'=' * 80}\nCOMPARISON\n{'=' * 80}")
    fixed = results[container.config.collection_fixed]
    sentence = results[container.config.collection_sentence]
    print(f"kb_fixed:    avg precision@3={fixed['avg_precision_at_3']:.3f}  avg recall@3={fixed['avg_recall_at_3']:.3f}")
    print(f"kb_sentence: avg precision@3={sentence['avg_precision_at_3']:.3f}  avg recall@3={sentence['avg_recall_at_3']:.3f}")


if __name__ == "__main__":
    main()

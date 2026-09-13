"""Optional demo: exercises the real (opt-in) HuggingFaceLLMProvider instead of
MOCK_LLM. Not part of the graded flow -- every acceptance criterion in the brief is
already demonstrated under MOCK_LLM by the other scripts in this directory.

Run with (downloads the model on first run):
    uv sync --extra real-llm
    LLM_PROVIDER=huggingface uv run python scripts/demo_real_llm.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import AppConfig
from src.container import ServiceContainer

QUERIES = [
    "What is the notice period for a Software Engineer?",
    "When is the referral bonus paid out?",
    "What is the best recipe for biryani?",
]

if __name__ == "__main__":
    config = AppConfig(llm_provider="huggingface")
    container = ServiceContainer(config)
    print(f"llm_provider = {container.config.llm_provider}  (model = {container.config.hf_llm_model})\n")

    for query in QUERIES:
        result = container.llm_provider.answer(query, collection=config.collection_sentence, top_k=3)
        print(f"Q: {query}")
        print(f"   top_similarity={result['top_similarity']:.3f}  grounded={result['grounded']}")
        print(f"   A: {result['answer']}\n")

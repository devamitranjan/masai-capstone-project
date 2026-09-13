"""Central configuration: every path, model name, and threshold that used to be a
scattered module-level constant now lives on one AppConfig, injected via
ServiceContainer instead of imported as a global."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

IDK_FALLBACK = (
    "I don't know. This question is outside the scope of the Naukri.com HR policy "
    "knowledge base I have access to -- please contact HR directly for this."
)


@dataclass(frozen=True)
class AppConfig:
    chroma_dir: str = "./chroma_db"
    embedding_model_name: str = "all-MiniLM-L6-v2"
    collection_fixed: str = "kb_fixed"
    collection_sentence: str = "kb_sentence"

    calibrated_threshold: float = 0.40
    idk_fallback: str = IDK_FALLBACK

    kb_dir: str = "knowledge_base"
    fixed_chunk_size: int = 150
    fixed_chunk_overlap: int = 30

    memory_file: Path = Path("data/conversation_memory.json")
    log_file: Path = Path("data/requests.log.jsonl")

    escalation_threshold: float = 0.65

    dataset_seed: int = 42
    dataset_num_records: int = 50

    llm_provider: str = field(default_factory=lambda: os.environ.get("LLM_PROVIDER", "mock"))
    hf_llm_model: str = field(default_factory=lambda: os.environ.get("HF_LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"))
    hf_llm_max_new_tokens: int = field(
        default_factory=lambda: int(os.environ.get("HF_LLM_MAX_NEW_TOKENS", "256"))
    )

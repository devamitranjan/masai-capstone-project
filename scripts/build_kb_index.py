"""Task 3: (re)builds both Chroma collections (kb_fixed, kb_sentence) from knowledge_base/*.md."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.container import ServiceContainer

if __name__ == "__main__":
    container = ServiceContainer.build_default()
    counts = container.index_builder.build()
    print(f"Indexed {counts['fixed']} fixed-size chunks into collection '{container.config.collection_fixed}'")
    print(f"Indexed {counts['sent']} sentence chunks into collection '{container.config.collection_sentence}'")

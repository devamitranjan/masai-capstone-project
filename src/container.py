"""Composition root: the ONE place concrete implementations are constructed and
wired together. Everywhere else (nodes, services, API routes) depends on
interfaces/injected instances, never on concrete classes or module-level globals.

`build_graph`/`run_agent` are thin convenience wrappers kept at module level so
existing call sites (`build_graph()`, `build_graph(enable_resilience=True)`,
`run_agent(query, thread_id=..., app=...)`) work unchanged; each builds its own
default ServiceContainer unless one is passed in explicitly.
"""

from __future__ import annotations

from src.agent.graph_factory import SupportAgentGraphBuilder
from src.agent.nodes import GuardrailInputNode, RagToolNode, RespondNode, RouterNode, StatusToolNode
from src.agent.router import IntentRouter
from src.agent.support_agent import SupportAgent
from src.chunking import DocumentChunker, FileSystemKnowledgeBaseRepository, FixedSizeChunkingStrategy, SentenceChunkingStrategy
from src.config import AppConfig
from src.guardrails import GroundednessGuardrail, GuardrailPipeline, PIIMaskingGuardrail, PromptInjectionGuardrail
from src.llm import HuggingFaceLLMProvider, MockLLMProvider, MockRagTriadJudge
from src.repositories import InMemoryJobApplicationRepository, JsonFileConversationStore
from src.retrieval import ChromaVectorStore, IndexBuilder, RetrievalService, SentenceTransformerEmbeddingModel
from src.scoring import EscalationScorer
from src.services import RequestLogger, StatusLookupService


class ServiceContainer:
    """Constructs every concrete implementation once and wires them together."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or AppConfig()

        self.embedder = SentenceTransformerEmbeddingModel(self.config.embedding_model_name)
        self.vector_store = ChromaVectorStore(self.config.chroma_dir)
        self.retrieval_service = RetrievalService(self.vector_store, self.embedder)

        self.kb_repository = FileSystemKnowledgeBaseRepository(self.config.kb_dir)
        self.chunker = DocumentChunker(self.kb_repository)
        self.chunking_strategies = [
            FixedSizeChunkingStrategy(self.config.fixed_chunk_size, self.config.fixed_chunk_overlap),
            SentenceChunkingStrategy(),
        ]
        self.index_builder = IndexBuilder(
            self.vector_store,
            self.embedder,
            self.chunker,
            self.chunking_strategies,
            collection_names={"fixed": self.config.collection_fixed, "sent": self.config.collection_sentence},
        )

        if self.config.llm_provider == "huggingface":
            self.llm_provider = HuggingFaceLLMProvider(
                self.retrieval_service,
                self.config.calibrated_threshold,
                self.config.idk_fallback,
                model_name=self.config.hf_llm_model,
                max_new_tokens=self.config.hf_llm_max_new_tokens,
            )
        else:
            self.llm_provider = MockLLMProvider(
                self.retrieval_service, self.config.calibrated_threshold, self.config.idk_fallback
            )
        self.llm_judge = MockRagTriadJudge(self.embedder)

        self.pii_guardrail = PIIMaskingGuardrail()
        self.input_guardrail_pipeline = GuardrailPipeline([self.pii_guardrail, PromptInjectionGuardrail()])
        self.groundedness_guardrail = GroundednessGuardrail(self.config.calibrated_threshold)

        self.job_application_repository = InMemoryJobApplicationRepository(
            self.config.dataset_seed, self.config.dataset_num_records
        )
        self.escalation_scorer = EscalationScorer(self.config.escalation_threshold)
        self.status_service = StatusLookupService(self.job_application_repository, self.escalation_scorer)

        self.conversation_store = JsonFileConversationStore(self.config.memory_file)
        self.request_logger = RequestLogger(self.config.log_file, self.pii_guardrail)

        self.graph_builder = SupportAgentGraphBuilder(
            guardrail_input=GuardrailInputNode(self.input_guardrail_pipeline),
            router=RouterNode(IntentRouter(self.conversation_store)),
            rag_tool=RagToolNode(self.llm_provider, self.groundedness_guardrail, self.config.collection_sentence),
            status_tool=StatusToolNode(self.status_service),
            respond=RespondNode(self.conversation_store),
        )

    def build_support_agent(self, **graph_kwargs) -> SupportAgent:
        return SupportAgent(self.graph_builder.build(**graph_kwargs))

    @classmethod
    def build_default(cls) -> "ServiceContainer":
        return cls(AppConfig())


def build_graph(
    checkpointer=None,
    interrupt_before: list[str] | None = None,
    enable_resilience: bool = False,
    container: ServiceContainer | None = None,
):
    """Convenience wrapper used by demos/tests that need the raw compiled graph
    (e.g. to call .ainvoke/.aget_state directly for checkpointing/resilience demos)."""
    container = container or ServiceContainer.build_default()
    return container.graph_builder.build(
        checkpointer=checkpointer, interrupt_before=interrupt_before, enable_resilience=enable_resilience
    )


async def run_agent(query: str, thread_id: str = "default", app=None) -> dict:
    """Convenience entrypoint used by demos and the FastAPI layer."""
    compiled = app or build_graph()
    return await SupportAgent(compiled).ask(query, thread_id)

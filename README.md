# Naukri.com Domain Support Agent (Recruitment & HR track)

**This submission completes the Naukri.com (Recruitment & HR) track** of the
Final Capstone — LangGraph domain support agent brief.

Everything in this repository runs deterministically under `MOCK_LLM` (no real LLM
call, no API keys, no network access required at inference time — only the one-time
local download of the `all-MiniLM-L6-v2` sentence-transformers embedding model).

## Dataset reproducibility (Part 1, Task 1)

Exact parameters used to deterministically regenerate the job-application dataset
(`InMemoryJobApplicationRepository` in `src/repositories.py`, driven by `dataset.py`):

| Parameter | Value |
|---|---|
| `SEED` | `42` |
| `NUM_RECORDS` | `50` |
| Category assignment | uniform random over the 5 given categories (round-robin seeded, then shuffled) |
| Status assignment | uniform random over the 5 given statuses (round-robin seeded, then shuffled) |
| `FLAGGED_PRIORITY_PROB` | `0.20` |
| `expected_salary_inr` range | ₹3,00,000 – ₹35,00,000/year (spans an entry-level Sales Associate/HR Executive hire up to a senior Software Engineer/Product Manager hire in the Indian market) |
| `days_since_created` | uniform integer in `[0, 30]` |

Measured output of `uv run python dataset.py` with these parameters:

```
Total records: 50
Count per category: Software Engineer=10, Data Analyst=10, Product Manager=10, HR Executive=10, Sales Associate=10
Count per status:   Applied=10, Screening=10, Interview Scheduled=10, Offered=10, Rejected=10
flagged_priority_review=True: 11/50 (22.0%)  -> within required 10%-30% band: True
80th percentile of days_since_created: 27 days
```

## Knowledge base (Part 1, Task 2)

12 documents in `knowledge_base/`, one per required topic, 4-5 sentences each: job
eligibility, interview scheduling, offer negotiation, background verification,
notice period, referral bonus, internal transfer, probation period, remote work,
diversity hiring, exit interview, applicant data retention.

## Chunking strategies and threshold calibration (Part 1, Tasks 3-5)

Two chunking strategies (`src/chunking.py`), each indexed into its own ChromaDB
collection (`src/retrieval.py`), both embedded with the same local
`all-MiniLM-L6-v2` model so the comparison below isolates chunking strategy:

| Strategy | Params | Collection | Chunk count |
|---|---|---|---|
| Fixed-size + overlap | 150 chars, 30 overlap | `kb_fixed` | 91 |
| Sentence-based | 1 sentence/chunk | `kb_sentence` | 60 |

**Threshold calibration** (measured by `scripts/demo_grounded_generation.py`,
top-1 cosine similarity):

| Query | fixed | sentence |
|---|---|---|
| "What is the notice period for a Software Engineer?" (in-scope) | 0.741 | 0.723 |
| "How long does background verification usually take?" (in-scope) | 0.605 | 0.656 |
| "When is the referral bonus paid out?" (in-scope) | 0.708 | 0.741 |
| "What is the capital of France?" (out-of-scope) | 0.109 | 0.022 |
| "How do I bake a chocolate cake?" (out-of-scope) | 0.089 | 0.081 |

In-scope cluster ≈ 0.60–0.77, out-of-scope cluster ≈ 0.02–0.11. We set
`calibrated_threshold = 0.40` (in `src/config.py`) — squarely in the observed gap,
not a tutorial preset. Demonstrated on 6 in-scope queries (all correctly grounded,
similarity 0.65–0.84) plus 1 deliberately out-of-scope query ("What is the best
recipe for biryani?", similarity 0.061) which correctly triggers the "I don't know"
fallback — see `transcripts/demo_grounded_generation.txt`.

**Precision@3 / Recall@3** (document-level, chunks deduped to parent doc before the
top-3 cutoff), same 6 queries, both collections — full per-query arithmetic in
`transcripts/demo_retrieval_eval.txt`:

| Collection | avg Precision@3 | avg Recall@3 |
|---|---|---|
| `kb_fixed` | 0.333 | 1.000 |
| `kb_sentence` | 0.333 | 1.000 |

Both strategies tie exactly: every query's single relevant document is correctly
retrieved in the top-3 for both collections (hence recall=1.000 on all 6 queries;
precision is mechanically capped at 1/3 since each query has exactly one relevant
document). **Recommendation: deploy `kb_sentence`.** Since Precision/Recall don't
differentiate here, we decide on chunk quality: fixed-size chunks frequently
truncate mid-sentence at the 150-character boundary (e.g. "...meet any stated
minimum years of" cut off in `01_eligibility_criteria`), producing answers that can
stop abruptly, whereas sentence-based chunks are always complete, self-contained
policy statements — and do so with 34% fewer chunks to index and search (60 vs 91).

## LangGraph agent (Part 2)

`src/agent/graph_factory.py`'s `SupportAgentGraphBuilder` builds one graph, reused by every demo and by Part 4's
resilience hardening:

```
START -> guardrail_input -> router --(policy_question)--> rag_tool ----> respond -> END
                                   --(status_lookup)----> status_tool -/
                                   --(blocked)---------------------------^
```

5 nodes (≥4 required), 1 genuine conditional edge (`route_from_intent`) routing on
query intent. Demonstrated on different queries in
`transcripts/demo_agent_routing.txt`.

- **Escalation score** (`src/services.StatusLookupService.check_status`, Task 6):
  `escalation_score = 0.5*flagged_priority_review + 0.5*min(days_since_created/30, 1.0)`,
  escalate when `score >= 0.65`. A flagged application clears the threshold once
  `days_since_created >= 9`, comfortably below the dataset's measured 80th
  percentile of 27 days — so a flagged application that's also meaningfully old
  (above-median age) reliably escalates, while an unflagged one never does by
  recency alone (full justification in the module docstring).
- **Memory** (`src/repositories.JsonFileConversationStore`, Task 8): JSON file `data/conversation_memory.json`
  keyed by `thread_id`. The status-lookup path actually reuses the last-mentioned
  `record_id` when a follow-up omits one. See
  `transcripts/demo_memory.txt` for both the continued-thread transcript and the
  separate fresh-thread transcript showing state correctly absent.
- **Structured output** (`src/schemas.py`, Task 9): pydantic `AgentResponse` ->
  JSON Schema via `.model_json_schema()`; every response is explicitly
  re-validated with `jsonschema.validate` in the `respond` node.
- **Guardrails** (`src/guardrails.py`, Task 10): input-side PII masking (fixed-format
  Indian phone number — the only fixed-format PII field per the brief; candidate
  name/salary/background-check results are free text and out of scope for a keyless
  masker), input-side prompt-injection detection, output-side groundedness refusal
  (reuses the Task 4 threshold). Each one demonstrated firing in
  `transcripts/demo_guardrails.txt`.

## FastAPI deployment, logging, evaluation (Part 3)

- `api.py` exposes `GET /health`, `POST /ask`, `POST /add-document` (Pydantic
  request/response models). Run: `uv run uvicorn api:app --host 0.0.0.0 --port 8001`.
- `src/services.RequestLogger` logs every request as one JSON-Lines entry
  (`data/requests.log.jsonl`) with a `trace_id` and `duration_ms`; the query text is
  masked with the same PII masker as the input guardrail before being written, so
  the phone number never reaches disk in the clear.
- `scripts/demo_rag_triad_eval.py` (Task 13) scores 15 queries — one per required KB
  topic (12) + 1 extra in-scope + 2 deliberately out-of-scope/edge — with a
  deterministic MOCK_LLM judge (`src/llm.MockRagTriadJudge`, cosine similarity over the same
  local embeddings):

  | Metric | Average across 15 queries |
  |---|---|
  | context_relevance | 0.6615 |
  | groundedness | 0.7463 |
  | answer_relevance | 0.6343 |

  Full per-query scores in `transcripts/demo_rag_triad_eval.txt` — both
  out-of-scope queries correctly score low (groundedness 0.161 and 0.402 vs.
  0.71–0.91 for in-scope queries).

## Optional: real LLM (opt-in, not used for grading)

Everything above runs — and every acceptance criterion is satisfied — under
`MOCK_LLM` alone, as required. On top of that, `src/llm.HuggingFaceLLMProvider`
wires in a real, local, keyless instruct model (default `Qwen/Qwen2.5-1.5B-Instruct`
via `transformers`) behind an env-var flag, implementing the same `LLMProvider`
interface as `MockLLMProvider` — it retrieves the same top-k context, respects the
same calibrated threshold/IDK fallback, and its output still passes through the
same output-side groundedness guardrail and structured-output schema. The
RAG-triad judge (Task 13) always stays `MockRagTriadJudge` regardless, since the
brief requires the graded evaluation itself to run under `MOCK_LLM`.

```bash
uv sync --extra real-llm                     # installs transformers/torch
LLM_PROVIDER=huggingface uv run python scripts/demo_real_llm.py
# or, to run the whole agent (API included) on the real model:
LLM_PROVIDER=huggingface uv run uvicorn api:app --host 0.0.0.0 --port 8001
```

`LLM_PROVIDER` defaults to `mock` (unset = graded behavior, unchanged). Override
the model/length with `HF_LLM_MODEL` / `HF_LLM_MAX_NEW_TOKENS`. `src/config.py`
loads `.env` via `python-dotenv` if present (e.g. for an `HF_TOKEN` to raise
Hugging Face Hub rate limits) — `.env` is gitignored and never required.

## Resilience & MCP (Part 4)

- **MCP** (`src/mcp_server.py`, Task 14): `fastmcp` server exposing
  `check_job_application_status`, served at `http://127.0.0.1:8000/mcp`. Separate
  client process `scripts/mcp_client_demo.py` calls it for 3 record IDs (2 found, 1
  not found) — see `transcripts/mcp_client_demo.txt`.
- **Checkpointing** (Task 15): `scripts/demo_checkpointing.py` uses
  `AsyncSqliteSaver` (`checkpoints.sqlite`) keyed by `thread_id`, with
  `interrupt_before=["respond"]`. The run executes `guardrail_input`, `router`,
  `rag_tool`, stops before `respond`, then resumes the SAME `thread_id` — the
  `[NODE EXECUTED]` trace prints show only `respond` firing on resume, proving the
  first three nodes were loaded from the checkpoint, not re-run. See
  `transcripts/demo_checkpointing.txt`.
- **Timeouts & retries** (Task 16): `scripts/demo_resilience.py`, built on
  LangGraph's native `RetryPolicy`/`TimeoutPolicy`:
  - `status_tool` has `RetryPolicy(initial_interval=0.2, backoff_factor=2.0,
    max_interval=2.0, max_attempts=4, jitter=True)`; a reserved record_id
    (`APP-RETRYTEST`) simulates failing the first 2 calls (`ConnectionError`) before
    succeeding on the 3rd — recovers transparently.
  - `rag_tool` has a per-node `TimeoutPolicy(run_timeout=2.0)`; a reserved query
    sentinel simulates a 5-second call, which cleanly raises
    `langgraph.errors.NodeTimeoutError` (not a hang).
  - The whole graph invocation is wrapped in `asyncio.wait_for(..., timeout=2.0)` in
    the demo script; the same slow-query simulation (without the per-node policy
    this time) correctly raises `asyncio.TimeoutError`, cancelling the entire run.

  Full output in `transcripts/demo_resilience.txt`.

## Project structure

```
masai-capstone-project/
├── dataset.py                  # Task 1 (JOB_APPLICATIONS list; generation lives in src/repositories.py)
├── knowledge_base/*.md         # Task 2 (12 docs)
├── src/
│   ├── config.py                # AppConfig: every path/model name/threshold in one place
│   ├── container.py              # ServiceContainer: composition root wiring everything together
│   ├── chunking.py               # Task 3 (ChunkingStrategy: FixedSize/Sentence, DocumentChunker)
│   ├── retrieval.py               # Task 3-4 (EmbeddingModel/VectorStore adapters, RetrievalService, IndexBuilder)
│   ├── retrieval_eval.py          # Task 5 (RetrievalEvaluator)
│   ├── repositories.py             # Task 1/8 (JobApplicationRepository, ConversationStore)
│   ├── scoring.py                  # Task 6 (EscalationScorer)
│   ├── services.py                  # Task 6/12 (StatusLookupService, RequestLogger)
│   ├── schemas.py                    # Task 9
│   ├── guardrails.py                  # Task 10 (GuardrailCheck strategies + GuardrailPipeline)
│   ├── llm.py                          # Task 4/13 (LLMProvider/LLMJudge; Mock + optional HuggingFace impls)
│   ├── agent/                           # Task 7/9/10/15/16
│   │   ├── state.py                      # AgentState
│   │   ├── router.py                     # IntentRouter
│   │   ├── nodes.py                       # one node class per graph step, DI'd
│   │   ├── graph_factory.py                # SupportAgentGraphBuilder
│   │   └── support_agent.py                 # SupportAgent facade
│   └── mcp_server.py                    # Task 14
├── api.py                      # Task 11
├── scripts/                    # one runnable demo per task
└── transcripts/                # captured output of every demo script, as evidence
```

## Running everything

```bash
uv sync

# Part 1
uv run python dataset.py
uv run python scripts/build_kb_index.py
uv run python scripts/demo_grounded_generation.py
uv run python scripts/demo_retrieval_eval.py

# Part 2
uv run python scripts/demo_agent_routing.py
uv run python scripts/demo_memory.py
uv run python scripts/demo_guardrails.py

# Part 3
uv run uvicorn api:app --host 0.0.0.0 --port 8001   # separate terminal
uv run python scripts/demo_rag_triad_eval.py

# Part 4
uv run python -m src.mcp_server                      # separate terminal, serves :8000/mcp
uv run python scripts/mcp_client_demo.py
uv run python scripts/demo_checkpointing.py
uv run python scripts/demo_resilience.py
```

No environment variables or API keys are required; everything above runs fully
offline after the one-time embedding-model download.

## Verification checklist

What to look for at each step, in order, to confirm every acceptance criterion in
the brief actually holds (not just that the script exits 0).

**Part 1 — Dataset & RAG core**

1. `uv run python dataset.py` — every category count ≥3 (all show 10), every status
   count ≥1 (all show 10), flagged-priority % between 10–30 (shows 22.0%) with the
   band check printing `True`.
2. `ls knowledge_base/*.md | wc -l` → `12`; each file covers one required topic in
   2–5 original sentences.
3. `uv run python scripts/build_kb_index.py` — prints non-zero chunk counts for
   both `kb_fixed` and `kb_sentence` (two separate collections).
4. `uv run python scripts/demo_grounded_generation.py` — measured similarities for
   ≥3 in-scope + ≥2 out-of-scope queries are shown, the chosen threshold sits
   between the two clusters, ≥5 in-scope queries answer with `grounded=True`, and
   1 out-of-scope query triggers the IDK fallback.
5. `uv run python scripts/demo_retrieval_eval.py` — per-query Precision@3/Recall@3
   arithmetic shown for **both** collections on the same queries, plus averages
   and a numbers-cited recommendation.

**Part 2 — LangGraph agent**

6. `uv run python scripts/demo_agent_routing.py` — a policy question routes to
   `rag_tool`, a status query routes to `status_tool`, a blocked query routes
   straight to `respond`; `escalation_score` is a real number (e.g. `0.1667`), not
   a boolean.
7. `uv run python scripts/demo_memory.py` — turn 2 in the continued thread reuses
   the record_id from turn 1; the separate fresh-thread transcript shows empty
   history and "no record_id remembered."
8. Structured output (Task 9) is checked implicitly by every script above: each
   one calls `validate_agent_response`, which raises on schema failure — confirm
   none of the runs raised an exception.
9. `uv run python scripts/demo_guardrails.py` — three sections, each showing its
   flag firing: PII phone number masked, prompt-injection query blocked,
   groundedness refusal on an out-of-scope query.

**Part 3 — FastAPI, logging, evaluation**

10. `uv run uvicorn api:app --host 0.0.0.0 --port 8001`, then:
    ```bash
    curl http://127.0.0.1:8001/health
    curl -X POST http://127.0.0.1:8001/ask -H 'Content-Type: application/json' \
      -d '{"query":"What is the notice period for a Software Engineer?"}'
    ```
    Both endpoints return Pydantic-validated JSON.
11. `cat data/requests.log.jsonl` — one JSON line per request with `trace_id` and
    `duration_ms`; send a query containing a phone number and confirm the logged
    `query` field shows `[PHONE_REDACTED]`, never the raw number.
12. `uv run python scripts/demo_rag_triad_eval.py` — 15 rows (12 topics + 3 extra),
    3 scores per row, plus the 3 averages printed at the bottom.

**Part 4 — Resilience & interoperability**

13. `uv run python -m src.mcp_server` (separate terminal), then
    `uv run python scripts/mcp_client_demo.py` — standardized MCP response printed
    for ≥2 record IDs from a separate client process.
14. `uv run python scripts/demo_checkpointing.py` — the first run shows 3 nodes
    executing then stopping before `respond`; the resume section shows **only**
    `respond` executing, proving the earlier nodes were loaded from the checkpoint
    rather than re-run.
15. `uv run python scripts/demo_resilience.py` — (a) `status_tool` executes 3 times
    (2 simulated failures + 1 success) with no exception escaping; (b) the
    per-node timeout raises `NodeTimeoutError` cleanly; (c) the global timeout
    raises `asyncio.TimeoutError` cleanly.

**Optional — real LLM (not graded)**

16. `uv sync --extra real-llm && LLM_PROVIDER=huggingface uv run python
    scripts/demo_real_llm.py` — real model answers are grounded/refused the same
    way MOCK_LLM's are. Then re-run any Part 2 demo with `LLM_PROVIDER` unset and
    confirm it's unaffected (still defaults to `mock`).

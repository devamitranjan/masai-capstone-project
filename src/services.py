"""Application services that orchestrate repositories/guardrails for a single use
case each (SRP): StatusLookupService (Task 6 lookup) and RequestLogger (Task 12
structured, PII-safe request logging), exposed as a context manager so API routes
don't repeat trace-id/timing/error-logging boilerplate per endpoint.
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from src.guardrails import PIIMaskingGuardrail
from src.repositories import JobApplicationRepository
from src.scoring import EscalationScorer

RETRY_TEST_RECORD_ID = "APP-RETRYTEST"


class StatusLookupService:
    """Task 6 — looks up a job application and attaches a designed escalation_score.

    Raises:
        ConnectionError: when record_id == RETRY_TEST_RECORD_ID and this is one of
        the first 2 calls on THIS instance -- used to demonstrate the Task 16 retry
        policy on a transient failure. Real record lookups are never affected.
    """

    def __init__(self, repository: JobApplicationRepository, scorer: EscalationScorer) -> None:
        self._repository = repository
        self._scorer = scorer
        self._retry_test_call_count = 0

    def reset_retry_test_counter(self) -> None:
        self._retry_test_call_count = 0

    def check_status(self, record_id: str) -> dict:
        if record_id == RETRY_TEST_RECORD_ID:
            self._retry_test_call_count += 1
            if self._retry_test_call_count <= 2:
                raise ConnectionError(
                    f"Simulated transient failure (attempt {self._retry_test_call_count}/2) for retry-policy demo"
                )

        record = self._repository.find_by_id(record_id)
        if record is None:
            return {"record_id": record_id, "found": False}

        score = self._scorer.compute(record.flagged_priority_review, record.days_since_created)
        return {
            "record_id": record.record_id,
            "found": True,
            "status": record.status,
            "expected_salary_inr": record.expected_salary_inr,
            "days_since_created": record.days_since_created,
            "flagged_priority_review": record.flagged_priority_review,
            "escalation_score": score,
            "escalation_recommended": self._scorer.is_escalation_recommended(score),
        }


class RequestTrace:
    def __init__(self, trace_id: str) -> None:
        self.trace_id = trace_id
        self.extra: dict = {}

    def record(self, extra: dict) -> None:
        self.extra.update(extra)


class RequestLogger:
    """Task 12 — structured JSON-Lines request logging with a trace ID and timing.
    The logged query is masked by the SAME PIIMaskingGuardrail used on the input-side
    guardrail, so a fixed-format PII field never reaches the log file in the clear.
    """

    def __init__(self, log_file: Path, pii_guardrail: PIIMaskingGuardrail) -> None:
        self._log_file = log_file
        self._pii_guardrail = pii_guardrail

    def log(self, trace_id: str, endpoint: str, raw_query: str, start_time: float, extra: dict | None = None) -> dict:
        masked = self._pii_guardrail.apply(raw_query)
        entry = {
            "trace_id": trace_id,
            "endpoint": endpoint,
            "query": masked.text,
            "pii_masked": "pii_masked" in masked.flags,
            "duration_ms": round((time.time() - start_time) * 1000, 2),
            "timestamp": time.time(),
        }
        if extra:
            entry.update(extra)

        self._log_file.parent.mkdir(parents=True, exist_ok=True)
        with self._log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        return entry

    @contextmanager
    def trace(self, endpoint: str, raw_query: str):
        """Wraps one request: logs exactly once on exit, with an error extra if the
        block raised, else whatever the caller passed to trace.record(...)."""
        trace_id = str(uuid.uuid4())
        start_time = time.time()
        request_trace = RequestTrace(trace_id)
        try:
            yield request_trace
        except Exception as e:
            self.log(trace_id, endpoint, raw_query, start_time, extra={"error": str(e)})
            raise
        else:
            self.log(trace_id, endpoint, raw_query, start_time, extra=request_trace.extra or None)

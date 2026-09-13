"""Task 1 — deterministic job-application dataset.

The generation rule and reproducibility rationale (seed, category/status
guarantees, salary band, flagged-priority band) live on
InMemoryJobApplicationRepository in src/repositories.py, so the agent (via
ServiceContainer) and this module share one generator instead of two divergent
copies of the same logic.

`JOB_APPLICATIONS` below is the literal module-level list the brief asks for:
a list of >=40 dicts, one per generated record, each with exactly the required
fields (record_id, category, status, expected_salary_inr, days_since_created,
flagged_priority_review).
"""

from __future__ import annotations

from dataclasses import asdict

from src.config import AppConfig
from src.repositories import InMemoryJobApplicationRepository

_config = AppConfig()
_repository = InMemoryJobApplicationRepository(seed=_config.dataset_seed, num_records=_config.dataset_num_records)

JOB_APPLICATIONS: list[dict] = [asdict(record) for record in _repository.list_all()]


def main() -> None:
    _repository.print_report()


if __name__ == "__main__":
    main()

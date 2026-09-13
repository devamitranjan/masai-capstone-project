"""Repository pattern for the two datasets the agent reads: job applications
(Task 1) and per-thread conversation history (Task 8). Each is behind an ABC so the
backing store (in-memory/generated today) can become a real database later without
any caller (StatusLookupService, IntentRouter, RespondNode) changing.
"""

from __future__ import annotations

import json
import random
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class JobApplication:
    record_id: str
    category: str
    status: str
    expected_salary_inr: int
    days_since_created: int
    flagged_priority_review: bool


class JobApplicationRepository(ABC):
    @abstractmethod
    def find_by_id(self, record_id: str) -> JobApplication | None: ...

    @abstractmethod
    def list_all(self) -> list[JobApplication]: ...


class InMemoryJobApplicationRepository(JobApplicationRepository):
    """Deterministic job-application dataset generator for the Naukri.com
    (Recruitment & HR) capstone track.

    Design choices (reproducibility parameters, also documented in README.md):
      - CATEGORIES and STATUSES are drawn uniformly at random (equal weights).
      - FLAGGED_PRIORITY_PROB = 0.20 -> realized flagged_priority_review fraction is
        checked by report() to land inside the required [10%, 30%] band.
      - expected_salary_inr range: INR 3,00,000 - 35,00,000 per year, spanning an
        entry-level hire at the low end to a senior hire at the high end.
      - days_since_created: uniform integer in [0, 30].
    """

    CATEGORIES = ["Software Engineer", "Data Analyst", "Product Manager", "HR Executive", "Sales Associate"]
    STATUSES = ["Applied", "Screening", "Interview Scheduled", "Offered", "Rejected"]
    SALARY_MIN_INR = 300_000
    SALARY_MAX_INR = 3_500_000
    FLAGGED_PRIORITY_PROB = 0.20

    def __init__(self, seed: int = 42, num_records: int = 50) -> None:
        self._seed = seed
        self._records = self._generate(seed, num_records)
        self._by_id = {r.record_id: r for r in self._records}

    def _generate(self, seed: int, num_records: int) -> list[JobApplication]:
        rng = random.Random(seed)

        categories_cycle = [self.CATEGORIES[i % len(self.CATEGORIES)] for i in range(num_records)]
        statuses_cycle = [self.STATUSES[i % len(self.STATUSES)] for i in range(num_records)]
        rng.shuffle(categories_cycle)
        rng.shuffle(statuses_cycle)

        records = []
        for i in range(num_records):
            expected_salary_inr = rng.randint(self.SALARY_MIN_INR, self.SALARY_MAX_INR)
            expected_salary_inr = round(expected_salary_inr / 10_000) * 10_000

            records.append(
                JobApplication(
                    record_id=f"APP-{i + 1:04d}",
                    category=categories_cycle[i],
                    status=statuses_cycle[i],
                    expected_salary_inr=expected_salary_inr,
                    days_since_created=rng.randint(0, 30),
                    flagged_priority_review=rng.random() < self.FLAGGED_PRIORITY_PROB,
                )
            )
        return records

    def find_by_id(self, record_id: str) -> JobApplication | None:
        return self._by_id.get(record_id)

    def list_all(self) -> list[JobApplication]:
        return list(self._records)

    def report(self) -> dict:
        records = self._records
        category_counts = Counter(r.category for r in records)
        status_counts = Counter(r.status for r in records)
        flagged_count = sum(1 for r in records if r.flagged_priority_review)
        flagged_pct = 100.0 * flagged_count / len(records)

        days = sorted(r.days_since_created for r in records)
        p80_index = min(len(days) - 1, round(0.8 * (len(days) - 1)))

        return {
            "num_records": len(records),
            "category_counts": dict(category_counts),
            "status_counts": dict(status_counts),
            "flagged_count": flagged_count,
            "flagged_pct": flagged_pct,
            "p80_days_since_created": days[p80_index],
        }

    def print_report(self) -> None:
        r = self.report()
        print(f"Seed: {self._seed}")
        print(f"Total records: {r['num_records']}")
        print("\nCount per category (every given category must be >= 3):")
        for cat in self.CATEGORIES:
            print(f"  {cat:20s}: {r['category_counts'].get(cat, 0)}")
        print("\nCount per status (every given status must be >= 1):")
        for st in self.STATUSES:
            print(f"  {st:20s}: {r['status_counts'].get(st, 0)}")
        print(
            f"\nflagged_priority_review=True: {r['flagged_count']}/{r['num_records']} "
            f"({r['flagged_pct']:.1f}%)  [must be in 10%-30%]"
        )
        in_band = 10.0 <= r["flagged_pct"] <= 30.0
        print(f"  -> within required band: {in_band}")
        print(f"\n80th percentile of days_since_created: {r['p80_days_since_created']} days")
        print("  (used in src/scoring.py to justify the escalation_score threshold)")


class ConversationStore(ABC):
    @abstractmethod
    def get_history(self, thread_id: str) -> list[dict]: ...

    @abstractmethod
    def append_turn(self, thread_id: str, role: str, content: str, extra: dict | None = None) -> None: ...

    @abstractmethod
    def get_last_record_id(self, thread_id: str) -> str | None: ...

    @abstractmethod
    def reset_thread(self, thread_id: str) -> None: ...


class JsonFileConversationStore(ConversationStore):
    """Task 8 — Conversation memory persisted to a JSON file, keyed by thread_id."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def _load_all(self) -> dict:
        if not self._path.exists():
            return {}
        return json.loads(self._path.read_text(encoding="utf-8"))

    def _save_all(self, data: dict) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def get_history(self, thread_id: str) -> list[dict]:
        return self._load_all().get(thread_id, [])

    def append_turn(self, thread_id: str, role: str, content: str, extra: dict | None = None) -> None:
        data = self._load_all()
        turns = data.setdefault(thread_id, [])
        turn = {"role": role, "content": content}
        if extra:
            turn.update(extra)
        turns.append(turn)
        self._save_all(data)

    def get_last_record_id(self, thread_id: str) -> str | None:
        for turn in reversed(self.get_history(thread_id)):
            if turn.get("record_id"):
                return turn["record_id"]
        return None

    def reset_thread(self, thread_id: str) -> None:
        data = self._load_all()
        data.pop(thread_id, None)
        self._save_all(data)

    def delete_file(self) -> None:
        if self._path.exists():
            self._path.unlink()

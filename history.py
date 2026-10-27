"""Track and summarize optimization runs.

A lightweight recorder so you can compare how different objectives,
starting points, and search strategies perform.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from hill_climb import Result


@dataclass
class RunRecord:
    """A single optimization run and the settings that produced it."""

    objective: str
    mode: str
    x: float
    value: float
    iterations: int


@dataclass
class History:
    """Collects :class:`RunRecord` entries and offers simple summaries."""

    records: list[RunRecord] = field(default_factory=list)

    def add(self, objective: str, mode: str, result: Result) -> RunRecord:
        record = RunRecord(
            objective=objective,
            mode=mode,
            x=result.x,
            value=result.value,
            iterations=result.iterations,
        )
        self.records.append(record)
        return record

    def best(self) -> RunRecord | None:
        """Return the record with the highest value, or None if empty."""
        if not self.records:
            return None
        return max(self.records, key=lambda r: r.value)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps([asdict(r) for r in self.records], indent=indent)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "History":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(records=[RunRecord(**item) for item in data])

    def __len__(self) -> int:
        return len(self.records)

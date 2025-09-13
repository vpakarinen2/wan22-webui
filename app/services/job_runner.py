"""Job runner."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Job:
    id: str
    task: str
    args: list[str]
    status: str = "queued"
    output_path: Optional[str] = None
    log_path: Optional[str] = None
    error: Optional[str] = None


class JobRunner:
    def __init__(self) -> None:
        self._current: Optional[Job] = None

    def submit(self, job: Job) -> str:
        self._current = job
        return job.id

    def cancel_current(self) -> bool:
        return False

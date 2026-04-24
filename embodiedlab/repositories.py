"""Shared repository protocols used across the API and trainer services."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from embodiedlab.result_models import Progress, ResultStatus
    from embodiedlab.schemas import SubmitRequest


class SubmissionWriter(Protocol):
    """Write boundary for submission persistence."""

    def save(self, req: SubmitRequest) -> str:
        """Persist a new submission and return its ID."""


class SubmissionExistenceChecker(Protocol):
    """Read boundary for checking whether a submission exists."""

    def exists(self, submission_id: str) -> bool:
        """Return whether a submission exists."""


class SubmissionReader(Protocol):
    """Read boundary for fetching a submission payload."""

    def fetch(self, submission_id: str) -> dict[str, Any] | None:
        """Fetch a submission payload by ID."""


class ResultReader(Protocol):
    """Read boundary for result lookup."""

    def fetch(self, submission_id: str) -> dict[str, Any] | None:
        """Fetch a result payload if it exists."""


class ResultQueueWriter(Protocol):
    """Write boundary for queuing result tracking."""

    def create_queued(self, submission_id: str) -> None:
        """Create a queued result document for a submission."""


class ResultFailureWriter(Protocol):
    """Write boundary for failed result transitions."""

    def mark_failed(
        self,
        submission_id: str,
        progress: Progress,
        message: str,
    ) -> None:
        """Mark a result document as failed."""


class ResultUpdateWriter(Protocol):
    """Write boundary for partial result document updates."""

    def write_update(  # noqa: PLR0913
        self,
        submission_id: str,
        *,
        status: ResultStatus,
        progress: Progress,
        summary: dict[str, Any] | None = None,
        error: str | None = None,
        artifacts: dict[str, Any] | None = None,
    ) -> None:
        """Persist a partial result update."""

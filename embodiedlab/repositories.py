"""Shared repository protocols used across the API and trainer services."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from embodiedlab.result_models import Progress, ResultBundle, ResultStatus
    from embodiedlab.schemas import ScenarioBundle, SubmissionControl


class SubmissionWriter(Protocol):
    """Write boundary for submission persistence."""

    def save(self, scenario: ScenarioBundle, *, cancel_token_hash: str) -> str:
        """Persist a new submission and return its ID."""


class SubmissionExistenceChecker(Protocol):
    """Read boundary for checking whether a submission exists."""

    def exists(self, submission_id: str) -> bool:
        """Return whether a submission exists."""


class SubmissionReader(Protocol):
    """Read boundary for fetching a submission payload."""

    def fetch(self, submission_id: str) -> dict[str, Any] | None:
        """Fetch a submission payload by ID."""


class SubmissionControlReader(Protocol):
    """Read boundary for private submission control data."""

    def fetch_control(self, submission_id: str) -> SubmissionControl | None:
        """Fetch cancellation and execution control data."""


class SubmissionExecutionWriter(Protocol):
    """Write boundary for the Cloud Run execution assigned to a submission."""

    def set_execution_name(self, submission_id: str, execution_name: str) -> None:
        """Persist the exact Cloud Run Execution resource name."""


class ResultReader(Protocol):
    """Read boundary for result lookup."""

    def fetch(self, submission_id: str) -> dict[str, Any] | None:
        """Fetch a result payload if it exists."""


class ResultQueueWriter(Protocol):
    """Write boundary for queuing result tracking."""

    def create_queued(self, submission_id: str) -> None:
        """Create a queued result document for a submission."""


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
        result_bundle: dict[str, Any] | ResultBundle | None = None,
    ) -> None:
        """Persist a partial result update."""

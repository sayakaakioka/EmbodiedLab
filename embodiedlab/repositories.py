"""Shared repository protocols used across the API and trainer services."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from datetime import datetime

    from embodiedlab.result_models import Progress, ResultBundle, ResultStatus
    from embodiedlab.schemas import ScenarioBundle, SubmissionControl


class SubmissionConflictError(Exception):
    """Raised when an idempotency key is reused for a different request."""


class SubmissionRecoveryError(Exception):
    """Raised when an existing idempotent submission cannot be recovered."""


class SubmissionAcceptanceWriter(Protocol):
    """Atomic write boundary for a submission and its queued result."""

    def accept(
        self,
        scenario: ScenarioBundle,
        *,
        cancel_token_hash: str,
        total_steps: int,
        idempotency_key: str,
    ) -> str:
        """Atomically persist and return a submission with its initial result."""


class SubmissionReader(Protocol):
    """Read boundary for fetching a submission payload."""

    def fetch(self, submission_id: str) -> dict[str, Any] | None:
        """Fetch a submission payload by ID."""


class SubmissionControlReader(Protocol):
    """Read boundary for private submission control data."""

    def fetch_control(self, submission_id: str) -> SubmissionControl | None:
        """Fetch cancellation and execution control data."""


class SubmissionDispatchWriter(Protocol):
    """Write boundary for the server-owned dispatch lifecycle."""

    def claim_dispatch(self, submission_id: str) -> bool:
        """Claim a pending submission for one dispatch attempt."""

    def mark_dispatched(self, submission_id: str, execution_name: str) -> bool:
        """Persist an exact execution while dispatch remains unresolved."""

    def mark_dispatch_ambiguous(self, submission_id: str, message: str) -> bool:
        """Persist an unresolved dispatch while it is still dispatching."""

    def fail_dispatch_if_queued(
        self,
        submission_id: str,
        *,
        progress: Progress,
        error: str,
    ) -> bool:
        """Atomically fail dispatch control while its result is still queued."""


class SubmissionCancellationWriter(Protocol):
    """Write boundary for the server-owned cancellation lifecycle."""

    def cancel_pending_dispatch(
        self,
        submission_id: str,
        *,
        progress: Progress,
    ) -> dict[str, Any] | None:
        """Atomically cancel a pending dispatch and its queued result."""

    def claim_cancellation(
        self,
        submission_id: str,
        *,
        claimed_at: datetime,
        stale_before: datetime,
    ) -> str | None:
        """Claim a cancellation lease and return its fencing token."""

    def mark_cancellation_requested(
        self,
        submission_id: str,
        lease_token: str,
    ) -> bool:
        """Record that the exact execution cancellation RPC was attempted."""

    def release_cancellation(
        self,
        submission_id: str,
        lease_token: str,
        error: str,
    ) -> bool:
        """Release a definitively rejected cancellation intent."""


class SubmissionWorkflowRepository(
    SubmissionAcceptanceWriter,
    SubmissionControlReader,
    SubmissionDispatchWriter,
    Protocol,
):
    """Combined repository boundary for the server-owned submission workflow."""


class SubmissionControlStore(
    SubmissionControlReader,
    SubmissionCancellationWriter,
    Protocol,
):
    """Combined read and write boundary for private submission control."""


class ResultReader(Protocol):
    """Read boundary for result lookup."""

    def fetch(self, submission_id: str) -> dict[str, Any] | None:
        """Fetch a result payload if it exists."""


class ResultTransitionWriter(Protocol):
    """Compare-and-set boundary for API-owned result transitions."""

    def transition_if_status(  # noqa: PLR0913
        self,
        submission_id: str,
        *,
        expected_statuses: set[ResultStatus],
        status: ResultStatus,
        progress: Progress,
        summary: dict[str, Any] | None = None,
        error: str | None = None,
        result_bundle: dict[str, Any] | ResultBundle | None = None,
    ) -> dict[str, Any] | None:
        """Apply and return a result transition only from an expected status."""


class ResultStore(ResultReader, ResultTransitionWriter, Protocol):
    """Combined read and compare-and-set boundary for result lifecycle state."""


class ResultProgressTransitionWriter(Protocol):
    """Atomic status transition that preserves canonical progress counters."""

    def transition_status_preserving_progress(
        self,
        submission_id: str,
        *,
        expected_statuses: set[ResultStatus],
        status: ResultStatus,
        message: str,
        error: str | None = None,
    ) -> dict[str, Any] | None:
        """Change status and message without rolling progress counters back."""


class ApiResultStore(ResultStore, ResultProgressTransitionWriter, Protocol):
    """Result lifecycle boundary used by the API service."""

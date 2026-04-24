"""Factory functions for building Progress snapshots at each training phase."""

from __future__ import annotations

from embodiedlab.result_models import Progress
from embodiedlab.result_models import completed_progress as shared_completed_progress
from embodiedlab.result_models import failed_progress as shared_failed_progress
from embodiedlab.result_models import running_progress as shared_running_progress
from embodiedlab.result_models import starting_progress as shared_starting_progress


def failed_progress(message: str, total_steps: int = 0) -> Progress:
    """Return a failed-phase Progress with the given error message."""
    return shared_failed_progress(message, total_steps)


def starting_progress(total_steps: int) -> Progress:
    """Return a starting-phase Progress for a job with the given total step count."""
    return shared_starting_progress(total_steps)


def running_progress(total_steps: int) -> Progress:
    """Return a running-phase Progress for a job with the given total step count."""
    return shared_running_progress(total_steps)


def completed_progress(total_steps: int) -> Progress:
    """Return a completed-phase Progress with current_step equal to total_steps."""
    return shared_completed_progress(total_steps)

"""Logging helpers for the trainer job."""

from __future__ import annotations

import json
import sys


def log_trainer_event(
    event: str,
    *,
    submission_id: str,
    **fields: object,
) -> None:
    """Write a structured trainer event to stdout as a single JSON line."""
    payload = {
        "event": event,
        "submission_id": submission_id,
        **fields,
    }
    sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")

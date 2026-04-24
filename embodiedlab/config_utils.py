"""Shared helpers for loading required configuration from the environment."""

from __future__ import annotations

import os


def get_required_env(name: str) -> str:
    """Return an environment variable value or raise if it is missing."""
    value = os.environ.get(name)
    if not value:
        msg = f"{name} is not set"
        raise RuntimeError(msg)

    return value

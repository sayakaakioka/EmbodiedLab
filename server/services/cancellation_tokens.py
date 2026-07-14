"""Capability token helpers for cloud job cancellation."""

from __future__ import annotations

import hashlib
import secrets

CANCEL_TOKEN_BYTES = 32


def issue_cancel_token() -> str:
    """Return a new high-entropy URL-safe cancellation capability."""
    return secrets.token_urlsafe(CANCEL_TOKEN_BYTES)


def hash_cancel_token(cancel_token: str) -> str:
    """Return the SHA-256 digest persisted instead of the raw capability."""
    return hashlib.sha256(cancel_token.encode("utf-8")).hexdigest()


def verify_cancel_token(cancel_token: str, expected_hash: str) -> bool:
    """Compare a presented capability with its persisted digest."""
    return secrets.compare_digest(hash_cancel_token(cancel_token), expected_hash)

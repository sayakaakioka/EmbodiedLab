"""Capability token helpers for cloud job cancellation."""

import hashlib
import secrets


def hash_cancel_token(cancel_token: str) -> str:
    """Return the SHA-256 digest persisted instead of the raw capability."""
    return hashlib.sha256(cancel_token.encode("utf-8")).hexdigest()


def verify_cancel_token(cancel_token: str, expected_hash: str) -> bool:
    """Compare a presented capability with its persisted digest."""
    return secrets.compare_digest(hash_cancel_token(cancel_token), expected_hash)

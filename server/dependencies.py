"""FastAPI dependency providers for config and Firestore client."""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from google.cloud import firestore

from server.config import ServerConfig, load_server_config
from server.repositories import (
    FirestoreResultRepository,
    FirestoreSubmissionRepository,
)


@lru_cache(maxsize=1)
def get_config() -> ServerConfig:
    """Return the server configuration singleton."""
    return load_server_config()


@lru_cache(maxsize=1)
def _create_db() -> firestore.Client:
    return firestore.Client(database=get_config().db_id)


def get_db() -> firestore.Client:
    """Return the Firestore client singleton."""
    return _create_db()


def get_submission_repository(
    db: Annotated[firestore.Client, Depends(get_db)],
) -> FirestoreSubmissionRepository:
    """Return the submission repository bound to the shared Firestore client."""
    return FirestoreSubmissionRepository(db)


def get_result_repository(
    db: Annotated[firestore.Client, Depends(get_db)],
) -> FirestoreResultRepository:
    """Return the result repository bound to the shared Firestore client."""
    return FirestoreResultRepository(db)

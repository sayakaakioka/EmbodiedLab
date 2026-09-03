"""Wire models shared by EmbodiedLab API clients and services."""

from typing import Literal

from pydantic import BaseModel, Field

RECOVERY_VALUE_MIN_LENGTH = 32
RECOVERY_VALUE_MAX_LENGTH = 128
RECOVERY_VALUE_PATTERN = r"^[A-Za-z0-9_-]+$"


class SubmissionResponse(BaseModel):
    """Response returned after accepting a new submission."""

    status: Literal["accepted"]
    submission_id: str = Field(min_length=1)
    cancel_token: str = Field(
        min_length=RECOVERY_VALUE_MIN_LENGTH,
        max_length=RECOVERY_VALUE_MAX_LENGTH,
        pattern=RECOVERY_VALUE_PATTERN,
    )

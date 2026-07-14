"""Wire models shared by EmbodiedLab API clients and services."""

from typing import Literal

from pydantic import BaseModel, Field


class SubmissionResponse(BaseModel):
    """Response returned after accepting a new submission."""

    status: Literal["accepted"]
    submission_id: str = Field(min_length=1)
    cancel_token: str = Field(min_length=32)


class TrainingResponse(BaseModel):
    """Response returned after accepting a training request."""

    status: Literal["accepted"]
    submission_id: str = Field(min_length=1)

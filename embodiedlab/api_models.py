"""Wire models shared by EmbodiedLab API clients and services."""

from typing import Literal

from pydantic import BaseModel, Field


class SubmissionResponse(BaseModel):
    """Response returned after accepting a submission or training request."""

    status: Literal["accepted"]
    submission_id: str = Field(min_length=1)

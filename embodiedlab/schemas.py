"""Pydantic request/response schemas for the submission API."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

from pydantic import BaseModel, Field, model_validator

from embodiedlab.training.training_config import TrainingConfig
from embodiedlab.training.training_models import GridPosition

GridSizeValue = Annotated[int, Field(ge=2)]
GridSize = Annotated[list[GridSizeValue], Field(min_length=2, max_length=2)]


class Environment(BaseModel):
    """Grid-world layout: size, obstacles, goal, and robot starting position."""

    size: GridSize = Field(default_factory=lambda: [2, 2])
    obstacles: list[GridPosition] = Field(default_factory=list)
    goal: GridPosition = Field(default_factory=lambda: GridPosition(x=1, y=1))
    robot_start: GridPosition = Field(default_factory=lambda: GridPosition(x=0, y=0))

    @model_validator(mode="after")
    def validate_grid_layout(self) -> Environment:
        """Ensure all positions are inside the grid and do not overlap."""
        width, height = self.size
        positions = [
            ("goal", self.goal),
            ("robot_start", self.robot_start),
            *(
                (f"obstacles[{i}]", obstacle)
                for i, obstacle in enumerate(self.obstacles)
            ),
        ]

        for field_name, position in positions:
            if not (0 <= position.x < width and 0 <= position.y < height):
                msg = f"{field_name} must be inside the grid"
                raise ValueError(msg)

        obstacle_positions = {(obstacle.x, obstacle.y) for obstacle in self.obstacles}
        if (self.goal.x, self.goal.y) in obstacle_positions:
            msg = "goal must not overlap with obstacles"
            raise ValueError(msg)

        if (self.robot_start.x, self.robot_start.y) in obstacle_positions:
            msg = "robot_start must not overlap with obstacles"
            raise ValueError(msg)

        if self.goal == self.robot_start:
            msg = "goal and robot_start must not overlap"
            raise ValueError(msg)

        return self


class Robot(BaseModel):
    """Robot descriptor supplied with each submission."""

    type: str = Field(default="simple", min_length=1)


class SubmitRequest(BaseModel):
    """Top-level request body for POST /submissions."""

    environment: Environment = Field(default_factory=Environment)
    robot: Robot = Field(default_factory=Robot)
    training: TrainingConfig = Field(default_factory=TrainingConfig)


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(UTC).isoformat()


class SubmissionDocument(BaseModel):
    """Firestore document stored at submissions/{submission_id}."""

    submission_id: str
    created_at: str = Field(default_factory=utc_now_iso)
    environment: Environment
    robot: Robot
    training: TrainingConfig


def build_submission_document(submission_id: str, req: SubmitRequest) -> dict:
    """Return a Firestore-ready dict for a new submission."""
    document = SubmissionDocument(
        submission_id=submission_id,
        environment=req.environment,
        robot=req.robot,
        training=req.training,
    )
    return document.model_dump(mode="json")

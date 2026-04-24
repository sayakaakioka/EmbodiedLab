"""Data models representing the grid-world specification."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field


class GridPosition(BaseModel):
    """An (x, y) cell coordinate inside the grid."""

    model_config = ConfigDict(frozen=True)

    x: int = Field(ge=0)
    y: int = Field(ge=0)


@dataclass(frozen=True)
class GridWorldSpec:
    """Fully resolved grid-world specification passed to the training runner."""

    width: int
    height: int
    obstacles: frozenset[GridPosition]
    goal: GridPosition
    robot_start: GridPosition
    robot_type: str

from dataclasses import dataclass
from typing import FrozenSet

from pydantic import BaseModel, ConfigDict, Field


class GridPosition(BaseModel):
	model_config = ConfigDict(frozen=True)

	x: int = Field(ge=0)
	y: int = Field(ge=0)


@dataclass(frozen=True)
class GridWorldSpec:
	width: int
	height: int
	obstacles: FrozenSet[GridPosition]
	goal: GridPosition
	robot_start: GridPosition
	robot_type: str

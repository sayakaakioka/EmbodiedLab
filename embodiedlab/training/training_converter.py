"""Convert EnvForge scenario bundles into the current training runtime spec."""

from __future__ import annotations

from math import floor
from typing import Any

from embodiedlab.schemas import Position2D, ScenarioBundle
from embodiedlab.training.training_models import GridPosition, GridWorldSpec


def parse_scenario_bundle(
    submission: dict[str, Any] | ScenarioBundle,
) -> ScenarioBundle:
    """Parse a Firestore submission document or scenario payload."""
    if isinstance(submission, ScenarioBundle):
        return submission

    payload = submission.get("scenario", submission)
    return ScenarioBundle.model_validate(payload)


def _position_to_grid(position: Position2D) -> GridPosition:
    return GridPosition(
        x=max(0, floor(position.x)),
        y=max(0, floor(position.z)),
    )


def convert_submission_to_spec(
    submission: dict[str, Any] | ScenarioBundle,
) -> GridWorldSpec:
    """Convert a ScenarioBundle into the current internal training spec."""
    scenario = parse_scenario_bundle(submission)
    bounds = scenario.world.bounds
    width = max(2, floor(bounds.max.x - bounds.min.x))
    height = max(2, floor(bounds.max.z - bounds.min.z))

    obstacles = {
        _position_to_grid(obstacle.center)
        for obstacle in scenario.world.static_obstacles
    }

    return GridWorldSpec(
        width=width,
        height=height,
        obstacles=frozenset(obstacles),
        goal=_position_to_grid(scenario.world.goal.position),
        robot_start=_position_to_grid(scenario.robot.start_pose.position),
        robot_type=scenario.robot.type.value,
    )

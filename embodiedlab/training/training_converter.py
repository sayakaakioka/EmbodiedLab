"""Convert EnvForge scenario bundles into the current training runtime spec."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor
from typing import Any

from embodiedlab.schemas import Position2D, ScenarioBundle
from embodiedlab.training.training_models import GridPosition, GridWorldSpec


@dataclass(frozen=True)
class ScenarioRuntimeConversion:
    """Boundary metadata for the temporary ScenarioBundle-to-grid adapter."""

    source_coordinate_system: str
    runtime_coordinate_system: str
    coordinate_mapping: str
    omitted_contract_fields: tuple[str, ...]
    lossy: bool
    notes: tuple[str, ...]


def parse_scenario_bundle(
    submission: dict[str, Any] | ScenarioBundle,
) -> ScenarioBundle:
    """Parse a Firestore submission document or scenario payload."""
    if isinstance(submission, ScenarioBundle):
        return submission

    payload = submission.get("scenario", submission)
    return ScenarioBundle.model_validate(payload)


def describe_runtime_conversion(
    submission: dict[str, Any] | ScenarioBundle,
) -> ScenarioRuntimeConversion:
    """Describe the current adapter from continuous EnvForge space to grid cells."""
    scenario = parse_scenario_bundle(submission)
    return ScenarioRuntimeConversion(
        source_coordinate_system=scenario.world.coordinate_system.value,
        runtime_coordinate_system="grid_world_cells",
        coordinate_mapping="subtract_bounds_min_then_floor_envforge_xz_meters_to_non_negative_xy_cells",
        omitted_contract_fields=(
            "world.static_walls",
            "world.static_obstacles[].size",
            "world.static_obstacles[].rotation_y_degrees",
            "world.goal.radius",
            "robot.start_pose.rotation_y_degrees",
            "robot.action_space",
            "sensors",
            "reward.components",
            "training.max_episode_steps",
        ),
        lossy=True,
        notes=(
            "Continuous x/z meter positions subtract bounds.min, then floor "
            "to grid cell indices.",
            "Object sizes and rotation_y_degrees are not represented by the "
            "current grid runtime.",
            "Reward components remain declarative at the contract boundary but "
            "are not fully mapped to runtime reward logic yet.",
        ),
    )


def _position_to_grid(position: Position2D, origin: Position2D) -> GridPosition:
    return GridPosition(
        x=max(0, floor(position.x - origin.x)),
        y=max(0, floor(position.z - origin.z)),
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
        _position_to_grid(obstacle.center, bounds.min)
        for obstacle in scenario.world.static_obstacles
    }

    return GridWorldSpec(
        width=width,
        height=height,
        obstacles=frozenset(obstacles),
        goal=_position_to_grid(scenario.world.goal.position, bounds.min),
        robot_start=_position_to_grid(scenario.robot.start_pose.position, bounds.min),
        robot_type=scenario.robot.type.value,
        envforge_origin_x=bounds.min.x,
        envforge_origin_z=bounds.min.z,
    )

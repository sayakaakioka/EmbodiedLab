"""Lightweight validation and sampling policy for randomized starts."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import cos, hypot, radians, sin
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from embodiedlab.training.training_models import ContinuousNavigationSpec

MAX_RANDOM_START_ATTEMPTS = 512
MIN_RANDOM_START_VALID_AREA_RATIO = 0.08
RANDOM_START_BOUNDARY_INSET_METERS = 1.35
RANDOM_START_CLEARANCE_RADIUS_METERS = 0.65
RANDOM_START_VALIDATION_GRID_SIZE = 32


class RandomStartValidationError(ValueError):
    """Raised when a scenario cannot support reliable randomized starts."""


@dataclass(frozen=True)
class RandomStartArea:
    """Validated sampling bounds and one deterministic safe grid position."""

    min_x: float
    min_z: float
    max_x: float
    max_z: float
    valid_area_ratio: float
    safe_position: tuple[float, float]
    _spec: ContinuousNavigationSpec = field(repr=False)

    def contains(self, x: float, z: float) -> bool:
        """Return whether a robot center is inside the valid start region."""
        return (
            self.min_x <= x <= self.max_x
            and self.min_z <= z <= self.max_z
            and is_valid_random_start_position(self._spec, x, z)
        )


def build_random_start_area(spec: ContinuousNavigationSpec) -> RandomStartArea:
    """Estimate the usable area on a fixed grid and choose a safe point."""
    min_x = spec.bounds.min_x + RANDOM_START_BOUNDARY_INSET_METERS
    max_x = spec.bounds.max_x - RANDOM_START_BOUNDARY_INSET_METERS
    min_z = spec.bounds.min_z + RANDOM_START_BOUNDARY_INSET_METERS
    max_z = spec.bounds.max_z - RANDOM_START_BOUNDARY_INSET_METERS
    if min_x >= max_x or min_z >= max_z:
        message = (
            "randomized start sampling area is empty after applying the "
            f"{RANDOM_START_BOUNDARY_INSET_METERS:g} m boundary inset"
        )
        raise RandomStartValidationError(message)

    valid_positions = [
        (x, z)
        for z in _grid_axis(min_z, max_z)
        for x in _grid_axis(min_x, max_x)
        if is_valid_random_start_position(spec, x, z)
    ]
    sample_count = RANDOM_START_VALIDATION_GRID_SIZE**2
    valid_area_ratio = len(valid_positions) / sample_count
    if valid_area_ratio < MIN_RANDOM_START_VALID_AREA_RATIO:
        message = (
            "randomized start valid area must be at least "
            f"{MIN_RANDOM_START_VALID_AREA_RATIO:.0%} of the sampling area; "
            f"estimated={valid_area_ratio:.2%}"
        )
        raise RandomStartValidationError(message)

    center_x = (min_x + max_x) / 2.0
    center_z = (min_z + max_z) / 2.0
    safe_position = min(
        valid_positions,
        key=lambda position: (
            (position[0] - center_x) ** 2 + (position[1] - center_z) ** 2,
            position,
        ),
    )
    return RandomStartArea(
        min_x=min_x,
        min_z=min_z,
        max_x=max_x,
        max_z=max_z,
        valid_area_ratio=valid_area_ratio,
        safe_position=safe_position,
        _spec=spec,
    )


def is_valid_random_start_position(
    spec: ContinuousNavigationSpec,
    x: float,
    z: float,
) -> bool:
    """Apply the shared obstacle and goal clearance checks to one point."""
    if hypot(x - spec.goal.x, z - spec.goal.z) <= (
        spec.goal.radius + RANDOM_START_CLEARANCE_RADIUS_METERS
    ):
        return False

    for obstacle in spec.obstacles:
        angle = radians(-obstacle.rotation_y_degrees)
        angle_cos = cos(angle)
        angle_sin = sin(angle)
        translated_x = x - obstacle.center_x
        translated_z = z - obstacle.center_z
        local_x = translated_x * angle_cos - translated_z * angle_sin
        local_z = translated_x * angle_sin + translated_z * angle_cos
        collision_half_x = (obstacle.size_x / 2.0) + spec.robot_radius
        collision_half_z = (obstacle.size_z / 2.0) + spec.robot_radius
        distance_x = max(abs(local_x) - collision_half_x, 0.0)
        distance_z = max(abs(local_z) - collision_half_z, 0.0)
        if hypot(distance_x, distance_z) <= RANDOM_START_CLEARANCE_RADIUS_METERS:
            return False
    return True


def _grid_axis(minimum: float, maximum: float) -> tuple[float, ...]:
    step = (maximum - minimum) / RANDOM_START_VALIDATION_GRID_SIZE
    return tuple(
        minimum + (index + 0.5) * step
        for index in range(RANDOM_START_VALIDATION_GRID_SIZE)
    )

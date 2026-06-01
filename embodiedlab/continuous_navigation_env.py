"""Gymnasium-compatible continuous navigation environment."""

from __future__ import annotations

from math import atan2, ceil, cos, degrees, radians, sin
from typing import TYPE_CHECKING, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

if TYPE_CHECKING:
    from embodiedlab.training.training_models import (
        ContinuousBoxObstacle,
        ContinuousNavigationSpec,
    )

ROBOT_OBSERVATION_SIZE = 3
GOAL_OBSERVATION_SIZE = 3
ACTION_SIZE = 2
FORWARD_ACTION_INDEX = 0
TURN_ACTION_INDEX = 1
MOVEMENT_COLLISION_STEP_METERS = 0.005
RAY_STEP_METERS = 0.05
WIDE_ANGLE_DEGREES = 90.0
REAR_ANGLE_DEGREES = 150.0


class ContinuousNavigationEnv(gym.Env):
    """Continuous x/z navigation runtime for EnvForge scenario bundles."""

    metadata: ClassVar[dict] = {"render_modes": []}

    def __init__(
        self,
        spec: ContinuousNavigationSpec,
        max_steps: int = 512,
    ) -> None:
        """Initialise observation/action spaces and reset robot state."""
        super().__init__()
        self.spec = spec
        self.max_steps = max_steps
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            shape=(ACTION_SIZE,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict(
            {
                "robot": spaces.Box(
                    low=np.array(
                        [spec.bounds.min_x, spec.bounds.min_z, -180.0],
                        dtype=np.float32,
                    ),
                    high=np.array(
                        [spec.bounds.max_x, spec.bounds.max_z, 180.0],
                        dtype=np.float32,
                    ),
                    shape=(ROBOT_OBSERVATION_SIZE,),
                    dtype=np.float32,
                ),
                "goal": spaces.Box(
                    low=np.array(
                        [spec.bounds.min_x, spec.bounds.min_z, 0.0],
                        dtype=np.float32,
                    ),
                    high=np.array(
                        [spec.bounds.max_x, spec.bounds.max_z, np.inf],
                        dtype=np.float32,
                    ),
                    shape=(GOAL_OBSERVATION_SIZE,),
                    dtype=np.float32,
                ),
                "front_distance": spaces.Box(
                    low=np.array([0.0], dtype=np.float32),
                    high=np.array(
                        [spec.distance_sensor_range_meters],
                        dtype=np.float32,
                    ),
                    shape=(1,),
                    dtype=np.float32,
                ),
            },
        )
        self.robot_pos = np.array(
            [spec.robot_start.x, spec.robot_start.z],
            dtype=np.float32,
        )
        self.robot_rotation_y_degrees = spec.robot_start.rotation_y_degrees
        self.steps = 0

    def _goal_pos(self) -> np.ndarray:
        return np.array([self.spec.goal.x, self.spec.goal.z], dtype=np.float32)

    def _distance_to_goal(self) -> float:
        return float(np.linalg.norm(self.robot_pos - self._goal_pos()))

    def _heading_vector(self) -> np.ndarray:
        angle = radians(self.robot_rotation_y_degrees)
        return np.array([sin(angle), cos(angle)], dtype=np.float32)

    def _signed_angle_to_goal_degrees(self) -> float:
        delta = self._goal_pos() - self.robot_pos
        target_degrees = degrees(atan2(float(delta[0]), float(delta[1])))
        return self._normalise_angle(target_degrees - self.robot_rotation_y_degrees)

    @staticmethod
    def _normalise_rotation(rotation_y_degrees: float) -> float:
        return ((rotation_y_degrees + 180.0) % 360.0) - 180.0

    @staticmethod
    def _normalise_angle(angle_degrees: float) -> float:
        return ((angle_degrees + 180.0) % 360.0) - 180.0

    def _inside_bounds(self, position: np.ndarray) -> bool:
        return bool(
            self.spec.bounds.min_x <= position[0] <= self.spec.bounds.max_x
            and self.spec.bounds.min_z <= position[1] <= self.spec.bounds.max_z,
        )

    @staticmethod
    def _inside_obstacle(
        position: np.ndarray,
        obstacle: ContinuousBoxObstacle,
    ) -> bool:
        angle = radians(-obstacle.rotation_y_degrees)
        translated = position - np.array(
            [obstacle.center_x, obstacle.center_z],
            dtype=np.float32,
        )
        local_x = translated[0] * cos(angle) - translated[1] * sin(angle)
        local_z = translated[0] * sin(angle) + translated[1] * cos(angle)
        return bool(
            abs(local_x) <= obstacle.size_x / 2.0
            and abs(local_z) <= obstacle.size_z / 2.0,
        )

    def _collision_id(self, position: np.ndarray) -> str | None:
        if not self._inside_bounds(position):
            return "world_bounds"
        for obstacle in self.spec.obstacles:
            if self._inside_obstacle(position, obstacle):
                return obstacle.obstacle_id
        return None
    def _segment_collision_id(
        self,
        start: np.ndarray,
        end: np.ndarray,
    ) -> str | None:
        delta = end - start
        distance = float(np.linalg.norm(delta))
        if distance == 0.0:
            return self._collision_id(end)

        sample_count = max(1, ceil(distance / MOVEMENT_COLLISION_STEP_METERS))
        for sample_index in range(1, sample_count + 1):
            probe = start + delta * (sample_index / sample_count)
            collision_id = self._collision_id(probe)
            if collision_id is not None:
                return collision_id
        return None


    def _front_distance(self) -> float:
        heading = self._heading_vector()
        max_range = self.spec.distance_sensor_range_meters
        distance = 0.0
        while distance < max_range:
            distance = min(distance + RAY_STEP_METERS, max_range)
            probe = self.robot_pos + heading * distance
            if self._collision_id(probe) is not None:
                return round(distance, 6)
        return max_range

    def _get_obs(self) -> dict[str, np.ndarray]:
        return {
            "robot": np.array(
                [
                    self.robot_pos[0],
                    self.robot_pos[1],
                    self.robot_rotation_y_degrees,
                ],
                dtype=np.float32,
            ),
            "goal": np.array(
                [self.spec.goal.x, self.spec.goal.z, self.spec.goal.radius],
                dtype=np.float32,
            ),
            "front_distance": np.array([self._front_distance()], dtype=np.float32),
        }

    def _get_info(
        self,
        *,
        collision_id: str | None = None,
        distance_delta: float = 0.0,
    ) -> dict:
        return {
            "distance": self._distance_to_goal(),
            "distance_delta": distance_delta,
            "collision": collision_id is not None,
            "collision_id": collision_id,
            "front_distance": self._front_distance(),
        }

    def reset(  # type: ignore[override]
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,  # noqa: ARG002
    ) -> tuple[dict[str, np.ndarray], dict]:
        """Reset the environment to the scenario start pose."""
        super().reset(seed=seed)
        self.robot_pos = np.array(
            [self.spec.robot_start.x, self.spec.robot_start.z],
            dtype=np.float32,
        )
        self.robot_rotation_y_degrees = self._normalise_rotation(
            self.spec.robot_start.rotation_y_degrees,
        )
        self.steps = 0
        return self._get_obs(), self._get_info()

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict]:  # type: ignore[override]
        """Apply forward/turn action and return observation, reward, and flags."""
        previous_distance = self._distance_to_goal()
        clipped_action = np.clip(
            np.asarray(action, dtype=np.float32),
            self.action_space.low,
            self.action_space.high,
        )
        self.steps += 1
        self.robot_rotation_y_degrees = self._normalise_rotation(
            self.robot_rotation_y_degrees
            + float(clipped_action[TURN_ACTION_INDEX])
            * self.spec.turn_degrees_per_step,
        )
        next_pos = self.robot_pos + (
            self._heading_vector()
            * float(clipped_action[FORWARD_ACTION_INDEX])
            * self.spec.forward_step_meters
        )
        collision_id = self._segment_collision_id(self.robot_pos, next_pos)
        if collision_id is None:
            self.robot_pos = next_pos.astype(np.float32)

        distance = self._distance_to_goal()
        distance_delta = previous_distance - distance
        terminated = distance <= self.spec.goal.radius
        truncated = self.steps >= self.max_steps
        weights = self.spec.reward_weights
        reward = weights.step_penalty + weights.goal_progress * distance_delta
        forward = float(clipped_action[FORWARD_ACTION_INDEX])
        turn = float(clipped_action[TURN_ACTION_INDEX])
        moving = (
            abs(forward) > weights.movement_threshold
            or abs(turn) > weights.movement_threshold
        )
        if moving:
            reward += weights.movement_reward
        signed_angle_to_goal = self._signed_angle_to_goal_degrees()
        if abs(signed_angle_to_goal) > REAR_ANGLE_DEGREES:
            reward += weights.rear_angle_penalty
        elif abs(signed_angle_to_goal) > WIDE_ANGLE_DEGREES:
            reward += weights.wide_angle_penalty
        inactive = (
            abs(forward) <= weights.movement_threshold
            or abs(turn) <= weights.turn_activity_threshold
        )
        if inactive:
            reward += weights.inactive_penalty
        if collision_id is not None:
            reward += weights.collision_penalty
        if terminated:
            reward += weights.goal_reached

        return (
            self._get_obs(),
            float(reward),
            terminated,
            truncated,
            self._get_info(
                collision_id=collision_id,
                distance_delta=distance_delta,
            ),
        )

"""Gymnasium-compatible continuous navigation environment."""

from __future__ import annotations

from math import atan2, ceil, cos, degrees, radians, sin, sqrt
from typing import TYPE_CHECKING, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from embodiedlab.training.navigation_final_policy import (
    POLICY_ACTION_HIGH,
    POLICY_ACTION_LOW,
)

if TYPE_CHECKING:
    from embodiedlab.training.training_models import (
        ContinuousNavigationSpec,
    )

IMAGE_OBSERVATION_CHANNELS = 3
IMAGE_OBSERVATION_HEIGHT = 84
IMAGE_OBSERVATION_WIDTH = 112
NUMERIC_OBSERVATION_SIZE = 2
ACTION_SIZE = 2
FORWARD_ACTION_INDEX = 0
TURN_ACTION_INDEX = 1
MOVEMENT_COLLISION_STEP_METERS = 0.005
RAY_STEP_METERS = 0.05
WIDE_ANGLE_DEGREES = 90.0
REAR_ANGLE_DEGREES = 150.0
CAMERA_FOV_DEGREES = 70.0
CAMERA_NEAR_METERS = 0.05
MAX_RANDOM_START_ATTEMPTS = 256
RANDOM_START_CLEARANCE_RADIUS_METERS = 0.65
RANDOM_START_CLEARANCE_PROBE_COUNT = 16
RANDOM_START_BOUNDARY_INSET_METERS = 1.35


class ContinuousNavigationEnv(gym.Env):
    """Continuous x/z navigation runtime for EnvForge scenario bundles."""

    metadata: ClassVar[dict] = {"render_modes": []}

    def __init__(
        self,
        spec: ContinuousNavigationSpec,
        max_steps: int = 512,
        *,
        randomize_start: bool = False,
    ) -> None:
        """Initialise observation/action spaces and reset robot state."""
        super().__init__()
        self.spec = spec
        self.max_steps = max_steps
        self.randomize_start = randomize_start
        self.action_space = spaces.Box(
            low=np.array(
                [POLICY_ACTION_LOW, POLICY_ACTION_LOW],
                dtype=np.float32,
            ),
            high=np.array(
                [POLICY_ACTION_HIGH, POLICY_ACTION_HIGH],
                dtype=np.float32,
            ),
            shape=(ACTION_SIZE,),
            dtype=np.float32,
        )
        max_goal_distance = sqrt(
            (spec.bounds.max_x - spec.bounds.min_x) ** 2
            + (spec.bounds.max_z - spec.bounds.min_z) ** 2,
        )
        self.observation_space = spaces.Dict(
            {
                "obs_0": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(
                        IMAGE_OBSERVATION_CHANNELS,
                        IMAGE_OBSERVATION_HEIGHT,
                        IMAGE_OBSERVATION_WIDTH,
                    ),
                    dtype=np.float32,
                ),
                "obs_1": spaces.Box(
                    low=np.array([-180.0, 0.0], dtype=np.float32),
                    high=np.array([180.0, max_goal_distance], dtype=np.float32),
                    shape=(NUMERIC_OBSERVATION_SIZE,),
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
        self._camera_column_angle_offsets = self._build_camera_column_angle_offsets()
        self._obstacle_ids = tuple(obstacle.obstacle_id for obstacle in spec.obstacles)
        self._obstacle_center_x = np.array(
            [obstacle.center_x for obstacle in spec.obstacles],
            dtype=np.float32,
        )
        self._obstacle_center_z = np.array(
            [obstacle.center_z for obstacle in spec.obstacles],
            dtype=np.float32,
        )
        self._obstacle_half_x = np.array(
            [obstacle.size_x / 2.0 for obstacle in spec.obstacles],
            dtype=np.float32,
        )
        self._obstacle_half_z = np.array(
            [obstacle.size_z / 2.0 for obstacle in spec.obstacles],
            dtype=np.float32,
        )
        obstacle_angles = np.deg2rad(
            np.array(
                [-obstacle.rotation_y_degrees for obstacle in spec.obstacles],
                dtype=np.float32,
            ),
        )
        self._obstacle_cos = np.cos(obstacle_angles)
        self._obstacle_sin = np.sin(obstacle_angles)

    @staticmethod
    def _build_camera_column_angle_offsets() -> np.ndarray:
        column_ratios = (
            np.arange(IMAGE_OBSERVATION_WIDTH, dtype=np.float32)
            / max(1, IMAGE_OBSERVATION_WIDTH - 1)
            - 0.5
        )
        return (column_ratios * CAMERA_FOV_DEGREES).astype(np.float32)

    def _map_raw_action(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        raw_action = np.clip(
            np.asarray(action, dtype=np.float32),
            self.action_space.low,
            self.action_space.high,
        ).astype(np.float32)
        applied_action = np.array(
            [
                (float(raw_action[FORWARD_ACTION_INDEX]) + 1.0) * 0.5,
                float(raw_action[TURN_ACTION_INDEX]),
            ],
            dtype=np.float32,
        )
        return raw_action, applied_action

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

    def _collision_id(self, position: np.ndarray) -> str | None:
        if not self._inside_bounds(position):
            return "world_bounds"
        if len(self._obstacle_ids) == 0:
            return None
        x = float(position[0])
        z = float(position[1])
        translated_x = x - self._obstacle_center_x
        translated_z = z - self._obstacle_center_z
        local_x = translated_x * self._obstacle_cos - translated_z * self._obstacle_sin
        local_z = translated_x * self._obstacle_sin + translated_z * self._obstacle_cos
        hits = np.nonzero(
            (np.abs(local_x) <= self._obstacle_half_x)
            & (np.abs(local_z) <= self._obstacle_half_z),
        )[0]
        if len(hits) > 0:
            return self._obstacle_ids[int(hits[0])]
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
        sample_ratios = np.arange(1, sample_count + 1, dtype=np.float32) / sample_count
        probes = start[None, :] + delta[None, :] * sample_ratios[:, None]
        return self._first_collision_id_for_points(probes[:, 0], probes[:, 1])

    def _front_distance(self) -> float:
        heading = self._heading_vector()
        max_range = self.spec.distance_sensor_range_meters
        sample_count = max(1, ceil(max_range / RAY_STEP_METERS))
        distances = np.minimum(
            np.arange(1, sample_count + 1, dtype=np.float32) * RAY_STEP_METERS,
            max_range,
        )
        probe_x = self.robot_pos[0] + heading[0] * distances
        probe_z = self.robot_pos[1] + heading[1] * distances
        collision_indices = np.nonzero(self._collision_mask(probe_x, probe_z))[0]
        if len(collision_indices) > 0:
            return round(float(distances[int(collision_indices[0])]), 6)
        return max_range

    def _get_obs(self) -> dict[str, np.ndarray]:
        return {
            "obs_0": self._render_segmentation_observation(),
            "obs_1": np.array(
                [
                    self._signed_angle_to_goal_degrees(),
                    self._distance_to_goal(),
                ],
                dtype=np.float32,
            ),
        }

    def _render_segmentation_observation(self) -> np.ndarray:
        image = np.zeros(
            (
                IMAGE_OBSERVATION_CHANNELS,
                IMAGE_OBSERVATION_HEIGHT,
                IMAGE_OBSERVATION_WIDTH,
            ),
            dtype=np.float32,
        )
        distances = self._camera_row_distances()
        ray_degrees = self.robot_rotation_y_degrees + self._camera_column_angle_offsets
        ray_radians = np.deg2rad(ray_degrees)
        probe_x = self.robot_pos[0] + np.sin(ray_radians)[None, :] * distances[:, None]
        probe_z = self.robot_pos[1] + np.cos(ray_radians)[None, :] * distances[:, None]
        collision_mask = self._collision_mask(probe_x, probe_z)
        image[1, :, :] = np.logical_not(collision_mask)
        image[2, :, :] = collision_mask
        return image

    def _camera_row_distances(self) -> np.ndarray:
        row_ratios = 1.0 - (
            np.arange(IMAGE_OBSERVATION_HEIGHT, dtype=np.float32)
            / max(1, IMAGE_OBSERVATION_HEIGHT - 1)
        )
        return (
            CAMERA_NEAR_METERS
            + row_ratios * (self.spec.distance_sensor_range_meters - CAMERA_NEAR_METERS)
        ).astype(np.float32)

    def _valid_random_start_position(self, position: np.ndarray) -> bool:
        return (
            self._clearance_collision_id(
                position,
                RANDOM_START_CLEARANCE_RADIUS_METERS,
            )
            is None
            and float(np.linalg.norm(position - self._goal_pos()))
            > self.spec.goal.radius + RANDOM_START_CLEARANCE_RADIUS_METERS
        )

    def _clearance_collision_id(
        self,
        position: np.ndarray,
        clearance_radius: float,
    ) -> str | None:
        angles = np.linspace(
            0.0,
            2.0 * np.pi,
            RANDOM_START_CLEARANCE_PROBE_COUNT,
            endpoint=False,
            dtype=np.float32,
        )
        probe_x = position[0] + np.cos(angles) * clearance_radius
        probe_z = position[1] + np.sin(angles) * clearance_radius
        probe_x = np.concatenate([np.asarray([position[0]], dtype=np.float32), probe_x])
        probe_z = np.concatenate([np.asarray([position[1]], dtype=np.float32), probe_z])
        return self._first_collision_id_for_points(probe_x, probe_z)

    def _sample_random_start(self) -> tuple[np.ndarray, float]:
        min_x = self.spec.bounds.min_x + RANDOM_START_BOUNDARY_INSET_METERS
        max_x = self.spec.bounds.max_x - RANDOM_START_BOUNDARY_INSET_METERS
        min_z = self.spec.bounds.min_z + RANDOM_START_BOUNDARY_INSET_METERS
        max_z = self.spec.bounds.max_z - RANDOM_START_BOUNDARY_INSET_METERS
        if min_x >= max_x or min_z >= max_z:
            min_x = self.spec.bounds.min_x
            max_x = self.spec.bounds.max_x
            min_z = self.spec.bounds.min_z
            max_z = self.spec.bounds.max_z

        for _attempt in range(MAX_RANDOM_START_ATTEMPTS):
            position = np.array(
                [
                    self.np_random.uniform(min_x, max_x),
                    self.np_random.uniform(min_z, max_z),
                ],
                dtype=np.float32,
            )
            if self._valid_random_start_position(position):
                rotation_y_degrees = float(self.np_random.uniform(-180.0, 180.0))
                return position, rotation_y_degrees

        return (
            np.array(
                [self.spec.robot_start.x, self.spec.robot_start.z],
                dtype=np.float32,
            ),
            self.spec.robot_start.rotation_y_degrees,
        )

    def _collision_mask(
        self,
        probe_x: np.ndarray,
        probe_z: np.ndarray,
    ) -> np.ndarray:
        mask = (
            (probe_x < self.spec.bounds.min_x)
            | (probe_x > self.spec.bounds.max_x)
            | (probe_z < self.spec.bounds.min_z)
            | (probe_z > self.spec.bounds.max_z)
        )
        if len(self._obstacle_ids) == 0:
            return mask
        for index in range(len(self._obstacle_ids)):
            translated_x = probe_x - self._obstacle_center_x[index]
            translated_z = probe_z - self._obstacle_center_z[index]
            local_x = (
                translated_x * self._obstacle_cos[index]
                - translated_z * self._obstacle_sin[index]
            )
            local_z = (
                translated_x * self._obstacle_sin[index]
                + translated_z * self._obstacle_cos[index]
            )
            mask |= (np.abs(local_x) <= self._obstacle_half_x[index]) & (
                np.abs(local_z) <= self._obstacle_half_z[index]
            )
        return mask

    def _first_collision_id_for_points(
        self,
        probe_x: np.ndarray,
        probe_z: np.ndarray,
    ) -> str | None:
        out_of_bounds = (
            (probe_x < self.spec.bounds.min_x)
            | (probe_x > self.spec.bounds.max_x)
            | (probe_z < self.spec.bounds.min_z)
            | (probe_z > self.spec.bounds.max_z)
        )
        if len(self._obstacle_ids) == 0:
            return "world_bounds" if np.any(out_of_bounds) else None

        translated_x = probe_x[:, None] - self._obstacle_center_x
        translated_z = probe_z[:, None] - self._obstacle_center_z
        local_x = translated_x * self._obstacle_cos - translated_z * self._obstacle_sin
        local_z = translated_x * self._obstacle_sin + translated_z * self._obstacle_cos
        obstacle_mask = (np.abs(local_x) <= self._obstacle_half_x) & (
            np.abs(local_z) <= self._obstacle_half_z
        )
        for point_index in range(len(probe_x)):
            if out_of_bounds[point_index]:
                return "world_bounds"
            obstacle_indices = np.nonzero(obstacle_mask[point_index])[0]
            if len(obstacle_indices) > 0:
                return self._obstacle_ids[int(obstacle_indices[0])]
        return None

    def _get_info(
        self,
        *,
        collision_id: str | None = None,
        distance_delta: float = 0.0,
        raw_action: np.ndarray | None = None,
        applied_action: np.ndarray | None = None,
        reward_components: list[dict[str, float | str]] | None = None,
    ) -> dict:
        info = {
            "distance": self._distance_to_goal(),
            "distance_delta": distance_delta,
            "collision": collision_id is not None,
            "collision_id": collision_id,
            "front_distance": self._front_distance(),
            "robot_x": float(self.robot_pos[0]),
            "robot_z": float(self.robot_pos[1]),
            "robot_rotation_y_degrees": float(self.robot_rotation_y_degrees),
            "goal_angle_degrees": self._signed_angle_to_goal_degrees(),
            "reward_components": reward_components or [],
        }
        if raw_action is not None and applied_action is not None:
            info.update(
                {
                    "raw_forward": float(raw_action[FORWARD_ACTION_INDEX]),
                    "raw_turn": float(raw_action[TURN_ACTION_INDEX]),
                    "applied_forward": float(applied_action[FORWARD_ACTION_INDEX]),
                    "applied_turn": float(applied_action[TURN_ACTION_INDEX]),
                },
            )
        return info

    def _reward_components(
        self,
        *,
        distance_delta: float,
        applied_forward: float,
        collision_id: str | None,
        goal_reached: bool,
    ) -> list[dict[str, float | str]]:
        weights = self.spec.reward_weights
        components: list[dict[str, float | str]] = [
            {"name": "step_penalty", "value": weights.step_penalty},
        ]
        if distance_delta > 0.0:
            components.append(
                {"name": "goal_progress", "value": weights.goal_progress},
            )

        signed_angle_to_goal = self._signed_angle_to_goal_degrees()
        if abs(signed_angle_to_goal) > REAR_ANGLE_DEGREES:
            components.append(
                {"name": "rear_angle_penalty", "value": weights.rear_angle_penalty},
            )
        elif abs(signed_angle_to_goal) > WIDE_ANGLE_DEGREES:
            components.append(
                {"name": "wide_angle_penalty", "value": weights.wide_angle_penalty},
            )

        if abs(applied_forward) <= weights.movement_threshold:
            components.append(
                {"name": "inactive_penalty", "value": weights.inactive_penalty},
            )
        if collision_id is not None:
            components.append(
                {"name": "collision_penalty", "value": weights.collision_penalty},
            )
        if goal_reached:
            components.append(
                {"name": "goal_reached", "value": weights.goal_reached},
            )
        return components

    def reset(  # type: ignore[override]
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,  # noqa: ARG002
    ) -> tuple[dict[str, np.ndarray], dict]:
        """Reset the environment to the scenario start pose."""
        super().reset(seed=seed)
        if self.randomize_start:
            position, rotation_y_degrees = self._sample_random_start()
            self.robot_pos = position
            self.robot_rotation_y_degrees = self._normalise_rotation(
                rotation_y_degrees,
            )
        else:
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
        raw_action, applied_action = self._map_raw_action(action)
        self.steps += 1
        self.robot_rotation_y_degrees = self._normalise_rotation(
            self.robot_rotation_y_degrees
            + float(applied_action[TURN_ACTION_INDEX])
            * self.spec.turn_degrees_per_step,
        )
        next_pos = self.robot_pos + (
            self._heading_vector()
            * float(applied_action[FORWARD_ACTION_INDEX])
            * self.spec.forward_step_meters
        )
        collision_id = self._segment_collision_id(self.robot_pos, next_pos)
        if collision_id is None:
            self.robot_pos = next_pos.astype(np.float32)

        distance = self._distance_to_goal()
        distance_delta = previous_distance - distance
        goal_reached = distance <= self.spec.goal.radius
        terminated = goal_reached or collision_id is not None
        truncated = self.steps >= self.max_steps
        reward_components = self._reward_components(
            distance_delta=distance_delta,
            applied_forward=float(applied_action[FORWARD_ACTION_INDEX]),
            collision_id=collision_id,
            goal_reached=goal_reached,
        )
        reward = sum(float(component["value"]) for component in reward_components)

        return (
            self._get_obs(),
            float(reward),
            terminated,
            truncated,
            self._get_info(
                collision_id=collision_id,
                distance_delta=distance_delta,
                raw_action=raw_action,
                applied_action=applied_action,
                reward_components=reward_components,
            ),
        )

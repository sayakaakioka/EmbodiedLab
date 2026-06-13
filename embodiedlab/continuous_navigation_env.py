"""Gymnasium-compatible continuous navigation environment."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, ceil, cos, degrees, radians, sin, sqrt, tan
from typing import TYPE_CHECKING, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from embodiedlab.training.navigation_final_policy import (
    POLICY_FORWARD_ACTION_HIGH,
    POLICY_FORWARD_ACTION_LOW,
    POLICY_TURN_ACTION_HIGH,
    POLICY_TURN_ACTION_LOW,
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
MIN_GOAL_PROGRESS_METERS = MOVEMENT_COLLISION_STEP_METERS
RAY_STEP_METERS = MOVEMENT_COLLISION_STEP_METERS
WIDE_ANGLE_DEGREES = 90.0
REAR_ANGLE_DEGREES = 150.0
BOUNDARY_WALL_THICKNESS_METERS = 0.02
RAY_EPSILON = 1e-6


@dataclass(frozen=True)
class CameraBox:
    """Extruded box used by the semantic camera ray caster."""

    center_x: float
    center_z: float
    half_x: float
    half_z: float
    height: float
    rotation_cos: float = 1.0
    rotation_sin: float = 0.0

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
                [POLICY_FORWARD_ACTION_LOW, POLICY_TURN_ACTION_LOW],
                dtype=np.float32,
            ),
            high=np.array(
                [POLICY_FORWARD_ACTION_HIGH, POLICY_TURN_ACTION_HIGH],
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
        self._obstacle_height = np.array(
            [obstacle.height for obstacle in spec.obstacles],
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
        self._camera_ray_directions = self._build_camera_ray_directions()

    def _map_raw_action(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        raw_action = np.clip(
            np.asarray(action, dtype=np.float32),
            self.action_space.low,
            self.action_space.high,
        ).astype(np.float32)
        applied_forward = 1.0 / (
            1.0 + np.exp(-float(raw_action[FORWARD_ACTION_INDEX]))
        )
        applied_turn = float(raw_action[TURN_ACTION_INDEX]) / POLICY_TURN_ACTION_HIGH
        applied_action = np.array(
            [applied_forward, applied_turn],
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

    def _ray_hits(self, ray_degrees: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        max_range = self.spec.distance_sensor_range_meters
        sample_count = max(1, ceil(max_range / RAY_STEP_METERS))
        distances = np.minimum(
            np.arange(1, sample_count + 1, dtype=np.float32) * RAY_STEP_METERS,
            max_range,
        )
        ray_radians = np.deg2rad(ray_degrees.astype(np.float32))
        probe_x = self.robot_pos[0] + np.sin(ray_radians)[None, :] * distances[:, None]
        probe_z = self.robot_pos[1] + np.cos(ray_radians)[None, :] * distances[:, None]
        collision_mask = self._collision_mask(probe_x, probe_z)
        has_hit = np.any(collision_mask, axis=0)
        first_indices = np.argmax(collision_mask, axis=0)
        hit_distances = np.full(ray_degrees.shape, max_range, dtype=np.float32)
        hit_distances[has_hit] = distances[first_indices[has_hit]]
        return hit_distances, has_hit

    def _front_distance(self) -> float:
        hit_distances, _has_hit = self._ray_hits(
            np.asarray([self.robot_rotation_y_degrees], dtype=np.float32),
        )
        return round(float(hit_distances[0]), 6)

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
        directions = self._world_camera_ray_directions()
        floor_distances = self._floor_intersection_distances(directions)
        blocked_distances = self._blocked_intersection_distances(directions)

        floor_mask = np.isfinite(floor_distances) & (
            floor_distances <= blocked_distances
        )
        blocked_mask = np.isfinite(blocked_distances) & (
            blocked_distances < floor_distances
        )
        background_mask = ~(floor_mask | blocked_mask)

        image[0, background_mask] = 1.0
        image[1, floor_mask] = 1.0
        image[2, blocked_mask] = 1.0
        return image

    def _build_camera_ray_directions(self) -> np.ndarray:
        camera = self.spec.camera
        if (
            camera.width != IMAGE_OBSERVATION_WIDTH
            or camera.height != IMAGE_OBSERVATION_HEIGHT
        ):
            msg = "camera dimensions must match the policy observation shape"
            raise ValueError(msg)

        vertical_fov_radians = radians(camera.vertical_fov_degrees)
        aspect = IMAGE_OBSERVATION_WIDTH / IMAGE_OBSERVATION_HEIGHT
        half_height = tan(vertical_fov_radians / 2.0)
        half_width = half_height * aspect
        x = (
            (np.arange(IMAGE_OBSERVATION_WIDTH, dtype=np.float32) + 0.5)
            / IMAGE_OBSERVATION_WIDTH
            * 2.0
            - 1.0
        ) * half_width
        y = (
            1.0
            - (np.arange(IMAGE_OBSERVATION_HEIGHT, dtype=np.float32) + 0.5)
            / IMAGE_OBSERVATION_HEIGHT
            * 2.0
        ) * half_height
        ray_x, ray_y = np.meshgrid(x, y)
        ray_z = np.ones_like(ray_x, dtype=np.float32)

        pitch = radians(camera.pitch_degrees)
        pitch_cos = cos(pitch)
        pitch_sin = sin(pitch)
        pitched_y = ray_y * pitch_cos - ray_z * pitch_sin
        pitched_z = ray_y * pitch_sin + ray_z * pitch_cos
        directions = np.stack((ray_x, pitched_y, pitched_z), axis=2)
        norms = np.linalg.norm(directions, axis=2, keepdims=True)
        return (directions / norms).astype(np.float32)

    def _world_camera_ray_directions(self) -> np.ndarray:
        yaw = radians(self.robot_rotation_y_degrees)
        yaw_cos = cos(yaw)
        yaw_sin = sin(yaw)
        local = self._camera_ray_directions
        world_x = local[:, :, 0] * yaw_cos + local[:, :, 2] * yaw_sin
        world_y = local[:, :, 1]
        world_z = -local[:, :, 0] * yaw_sin + local[:, :, 2] * yaw_cos
        return np.stack((world_x, world_y, world_z), axis=2).astype(np.float32)

    def _floor_intersection_distances(self, directions: np.ndarray) -> np.ndarray:
        camera = self.spec.camera
        dy = directions[:, :, 1]
        distances = np.full(dy.shape, np.inf, dtype=np.float32)
        downward = dy < -RAY_EPSILON
        floor_distances = -camera.mount_height_meters / dy[downward]
        valid = (
            (floor_distances >= camera.near_clip_meters)
            & (floor_distances <= camera.far_clip_meters)
        )
        downward_indices = np.nonzero(downward)
        distances[downward_indices[0][valid], downward_indices[1][valid]] = (
            floor_distances[valid]
        )
        return distances

    def _blocked_intersection_distances(self, directions: np.ndarray) -> np.ndarray:
        camera = self.spec.camera
        distances = np.full(
            (IMAGE_OBSERVATION_HEIGHT, IMAGE_OBSERVATION_WIDTH),
            np.inf,
            dtype=np.float32,
        )
        for index in range(len(self._obstacle_ids)):
            obstacle_distances = self._box_intersection_distances(
                directions,
                CameraBox(
                    center_x=float(self._obstacle_center_x[index]),
                    center_z=float(self._obstacle_center_z[index]),
                    half_x=float(self._obstacle_half_x[index]),
                    half_z=float(self._obstacle_half_z[index]),
                    height=float(self._obstacle_height[index]),
                    rotation_cos=float(self._obstacle_cos[index]),
                    rotation_sin=float(self._obstacle_sin[index]),
                ),
            )
            distances = np.minimum(distances, obstacle_distances)

        for boundary in self._boundary_boxes():
            boundary_distances = self._box_intersection_distances(
                directions,
                boundary,
            )
            distances = np.minimum(distances, boundary_distances)

        distances[
            (distances < camera.near_clip_meters)
            | (distances > camera.far_clip_meters)
        ] = np.inf
        return distances

    def _box_intersection_distances(
        self,
        directions: np.ndarray,
        box: CameraBox,
    ) -> np.ndarray:
        origin_x = float(self.robot_pos[0]) - box.center_x
        origin_z = float(self.robot_pos[1]) - box.center_z
        rotation_cos = box.rotation_cos
        rotation_sin = box.rotation_sin
        local_origin_x = origin_x * rotation_cos - origin_z * rotation_sin
        local_origin_z = origin_x * rotation_sin + origin_z * rotation_cos
        local_direction_x = (
            directions[:, :, 0] * rotation_cos
            - directions[:, :, 2] * rotation_sin
        )
        local_direction_z = (
            directions[:, :, 0] * rotation_sin
            + directions[:, :, 2] * rotation_cos
        )

        t_min_x, t_max_x, valid_x = self._axis_intersection_interval(
            local_origin_x,
            local_direction_x,
            -box.half_x,
            box.half_x,
        )
        t_min_y, t_max_y, valid_y = self._axis_intersection_interval(
            self.spec.camera.mount_height_meters,
            directions[:, :, 1],
            0.0,
            box.height,
        )
        t_min_z, t_max_z, valid_z = self._axis_intersection_interval(
            local_origin_z,
            local_direction_z,
            -box.half_z,
            box.half_z,
        )
        t_enter = np.maximum(np.maximum(t_min_x, t_min_y), t_min_z)
        t_exit = np.minimum(np.minimum(t_max_x, t_max_y), t_max_z)
        valid = valid_x & valid_y & valid_z & (t_exit >= t_enter)
        valid &= t_exit >= self.spec.camera.near_clip_meters
        distances = np.full(t_enter.shape, np.inf, dtype=np.float32)
        distances[valid] = np.maximum(
            t_enter[valid],
            self.spec.camera.near_clip_meters,
        )
        return distances

    @staticmethod
    def _axis_intersection_interval(
        origin: float,
        direction: np.ndarray,
        min_value: float,
        max_value: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        parallel = np.abs(direction) < RAY_EPSILON
        valid = ~(parallel & ((origin < min_value) | (origin > max_value)))
        safe_direction = np.where(parallel, 1.0, direction)
        t1 = (min_value - origin) / safe_direction
        t2 = (max_value - origin) / safe_direction
        t_min = np.minimum(t1, t2).astype(np.float32)
        t_max = np.maximum(t1, t2).astype(np.float32)
        t_min[parallel] = -np.inf
        t_max[parallel] = np.inf
        return t_min, t_max, valid

    def _boundary_boxes(self) -> tuple[CameraBox, ...]:
        bounds = self.spec.bounds
        span_x = bounds.max_x - bounds.min_x
        span_z = bounds.max_z - bounds.min_z
        center_x = (bounds.min_x + bounds.max_x) / 2.0
        center_z = (bounds.min_z + bounds.max_z) / 2.0
        height = max([2.0, *(float(value) for value in self._obstacle_height)])
        return (
            CameraBox(
                center_x=bounds.min_x,
                center_z=center_z,
                half_x=BOUNDARY_WALL_THICKNESS_METERS / 2.0,
                half_z=span_z / 2.0,
                height=height,
            ),
            CameraBox(
                center_x=bounds.max_x,
                center_z=center_z,
                half_x=BOUNDARY_WALL_THICKNESS_METERS / 2.0,
                half_z=span_z / 2.0,
                height=height,
            ),
            CameraBox(
                center_x=center_x,
                center_z=bounds.min_z,
                half_x=span_x / 2.0,
                half_z=BOUNDARY_WALL_THICKNESS_METERS / 2.0,
                height=height,
            ),
            CameraBox(
                center_x=center_x,
                center_z=bounds.max_z,
                half_x=span_x / 2.0,
                half_z=BOUNDARY_WALL_THICKNESS_METERS / 2.0,
                height=height,
            ),
        )

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
        if distance_delta >= MIN_GOAL_PROGRESS_METERS:
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

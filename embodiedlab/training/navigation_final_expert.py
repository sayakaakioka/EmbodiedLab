"""NavigationFinal expert pretraining helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from heapq import heappop, heappush
from math import atan2, degrees, hypot
from typing import TYPE_CHECKING

import numpy as np
import torch

from embodiedlab.continuous_navigation_env import ContinuousNavigationEnv

if TYPE_CHECKING:
    from stable_baselines3 import PPO

    from embodiedlab.training.training_models import ContinuousNavigationSpec

TrainingDiagnosticCallback = Callable[[str, dict[str, object]], None]

EXPERT_PRETRAIN_SAMPLES = 16384
EXPERT_PRETRAIN_BATCH_SIZE = 128
EXPERT_PRETRAIN_EPOCHS = 12
EXPERT_PRETRAIN_LEARNING_RATE = 1e-3
EXPERT_FORWARD_TARGET = 0.8
EXPERT_TURN_ANGLE_DENOMINATOR_DEGREES = 45.0
EXPERT_TURNING_FORWARD_TARGET = 0.25
EXPERT_WAYPOINT_REACHED_METERS = 0.45
EXPERT_PLANNER_GRID_RESOLUTION_METERS = 0.35
EXPERT_PLANNER_CLEARANCE_METERS = 0.35
EXPERT_SMOOTHING_CLEARANCE_METERS = 0.40
EXPERT_SMOOTHING_STEP_METERS = 0.05
MIN_WAYPOINTS_TO_SMOOTH = 3


@dataclass(frozen=True)
class _PlannerGrid:
    env: ContinuousNavigationEnv
    resolution: float = EXPERT_PLANNER_GRID_RESOLUTION_METERS

    def __post_init__(self) -> None:
        bounds = self.env.spec.bounds
        object.__setattr__(self, "min_x", bounds.min_x)
        object.__setattr__(self, "min_z", bounds.min_z)
        object.__setattr__(
            self,
            "width",
            round((bounds.max_x - bounds.min_x) / self.resolution) + 1,
        )
        object.__setattr__(
            self,
            "depth",
            round((bounds.max_z - bounds.min_z) / self.resolution) + 1,
        )

    def to_index(self, position: np.ndarray) -> tuple[int, int]:
        return (
            round((float(position[0]) - self.min_x) / self.resolution),
            round((float(position[1]) - self.min_z) / self.resolution),
        )

    def to_position(self, index: tuple[int, int]) -> np.ndarray:
        return np.array(
            [
                self.min_x + index[0] * self.resolution,
                self.min_z + index[1] * self.resolution,
            ],
            dtype=np.float32,
        )

    def is_free(self, index: tuple[int, int]) -> bool:
        if index[0] < 0 or index[0] >= self.width:
            return False
        if index[1] < 0 or index[1] >= self.depth:
            return False
        return (
            self.env._clearance_collision_id(  # noqa: SLF001
                self.to_position(index),
                EXPERT_PLANNER_CLEARANCE_METERS,
            )
            is None
        )

    def nearest_free_index(self, target: np.ndarray) -> tuple[int, int]:
        target_index = self.to_index(target)
        if self.is_free(target_index):
            return target_index

        candidates = (
            (float(np.linalg.norm(self.to_position((x, z)) - target)), (x, z))
            for x in range(self.width)
            for z in range(self.depth)
            if self.is_free((x, z))
        )
        return min(candidates, default=(0.0, target_index))[1]


def _expert_raw_action_from_goal_angle(goal_angle_degrees: float) -> list[float]:
    forward = (
        EXPERT_FORWARD_TARGET
        if abs(goal_angle_degrees) < EXPERT_TURN_ANGLE_DENOMINATOR_DEGREES
        else EXPERT_TURNING_FORWARD_TARGET
    )
    turn = float(
        np.clip(
            goal_angle_degrees / EXPERT_TURN_ANGLE_DENOMINATOR_DEGREES,
            -1.0,
            1.0,
        ),
    )
    return [forward * 2.0 - 1.0, turn]


def _angle_to_position_degrees(
    *,
    position: np.ndarray,
    target: np.ndarray,
    rotation_y_degrees: float,
) -> float:
    delta = target - position
    target_degrees = degrees(atan2(float(delta[0]), float(delta[1])))
    return ((target_degrees - rotation_y_degrees + 180.0) % 360.0) - 180.0


def _search_waypoint_indices(
    *,
    grid: _PlannerGrid,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> dict[tuple[int, int], tuple[int, int] | None]:
    frontier: list[tuple[float, tuple[int, int]]] = [(0.0, start)]
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    costs: dict[tuple[int, int], float] = {start: 0.0}
    neighbours = (
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1, 1),
        (1, -1),
        (-1, 1),
        (-1, -1),
    )

    while frontier:
        _priority, current = heappop(frontier)
        if current == goal:
            break
        for offset_x, offset_z in neighbours:
            neighbour = (current[0] + offset_x, current[1] + offset_z)
            if not grid.is_free(neighbour):
                continue
            new_cost = costs[current] + hypot(offset_x, offset_z)
            if neighbour in costs and costs[neighbour] <= new_cost:
                continue
            costs[neighbour] = new_cost
            came_from[neighbour] = current
            priority = new_cost + hypot(
                neighbour[0] - goal[0],
                neighbour[1] - goal[1],
            )
            heappush(frontier, (priority, neighbour))
    return came_from


def _reconstruct_waypoints(
    *,
    grid: _PlannerGrid,
    came_from: dict[tuple[int, int], tuple[int, int] | None],
    start_position: np.ndarray,
    goal_position: np.ndarray,
    goal: tuple[int, int],
) -> list[np.ndarray]:
    if goal not in came_from:
        return [start_position.copy(), goal_position]

    reversed_path: list[np.ndarray] = []
    current: tuple[int, int] | None = goal
    while current is not None:
        reversed_path.append(grid.to_position(current))
        current = came_from[current]
    return list(reversed(reversed_path))


def _segment_has_clearance(
    *,
    env: ContinuousNavigationEnv,
    start: np.ndarray,
    end: np.ndarray,
    clearance_meters: float,
) -> bool:
    distance = float(np.linalg.norm(end - start))
    checks = max(2, int(np.ceil(distance / EXPERT_SMOOTHING_STEP_METERS)) + 1)
    for weight in np.linspace(0.0, 1.0, checks):
        position = start + (end - start) * weight
        if env._clearance_collision_id(position, clearance_meters) is not None:  # noqa: SLF001
            return False
    return True


def _smooth_waypoints(
    *,
    env: ContinuousNavigationEnv,
    waypoints: list[np.ndarray],
) -> list[np.ndarray]:
    if len(waypoints) < MIN_WAYPOINTS_TO_SMOOTH:
        return waypoints

    smoothed = [waypoints[0]]
    index = 0
    while index < len(waypoints) - 1:
        next_index = len(waypoints) - 1
        while next_index > index + 1:
            if _segment_has_clearance(
                env=env,
                start=waypoints[index],
                end=waypoints[next_index],
                clearance_meters=EXPERT_SMOOTHING_CLEARANCE_METERS,
            ):
                break
            next_index -= 1
        smoothed.append(waypoints[next_index])
        index = next_index
    return smoothed


def _plan_expert_waypoints(env: ContinuousNavigationEnv) -> list[np.ndarray]:
    grid = _PlannerGrid(env)
    start = grid.to_index(env.robot_pos)
    goal_position = env._goal_pos()  # noqa: SLF001
    goal = grid.nearest_free_index(goal_position)
    came_from = _search_waypoint_indices(grid=grid, start=start, goal=goal)
    waypoints = _reconstruct_waypoints(
        grid=grid,
        came_from=came_from,
        start_position=env.robot_pos,
        goal_position=goal_position,
        goal=goal,
    )
    return _smooth_waypoints(env=env, waypoints=waypoints)


def _expert_rollout_samples(
    *,
    env: ContinuousNavigationEnv,
    seed: int,
) -> list[tuple[dict[str, np.ndarray], list[float]]]:
    obs, info = env.reset(seed=seed)
    waypoints = _plan_expert_waypoints(env)
    waypoint_index = 1 if len(waypoints) > 1 else 0
    samples: list[tuple[dict[str, np.ndarray], list[float]]] = []
    for _step in range(env.max_steps):
        if waypoints:
            while (
                waypoint_index < len(waypoints) - 1
                and float(np.linalg.norm(env.robot_pos - waypoints[waypoint_index]))
                < EXPERT_WAYPOINT_REACHED_METERS
            ):
                waypoint_index += 1
            goal_angle_degrees = _angle_to_position_degrees(
                position=env.robot_pos,
                target=waypoints[waypoint_index],
                rotation_y_degrees=env.robot_rotation_y_degrees,
            )
        else:
            goal_angle_degrees = float(info["goal_angle_degrees"])
        target_action = _expert_raw_action_from_goal_angle(goal_angle_degrees)
        samples.append(
            (
                {
                    "obs_0": obs["obs_0"].copy(),
                    "obs_1": obs["obs_1"].copy(),
                },
                target_action,
            ),
        )
        obs, _reward, terminated, truncated, info = env.step(
            np.asarray(target_action, dtype=np.float32),
        )
        if terminated or truncated:
            break
    return samples


def _emit_training_diagnostic(
    diagnostic_callback: TrainingDiagnosticCallback | None,
    event: str,
    **fields: object,
) -> None:
    if diagnostic_callback is not None:
        diagnostic_callback(event, fields)


def pretrain_navigation_final_policy(
    *,
    model: PPO,
    spec: ContinuousNavigationSpec,
    diagnostic_callback: TrainingDiagnosticCallback | None = None,
) -> None:
    """Warm-start NavigationFinal policy with obstacle-aware expert rollouts."""
    _emit_training_diagnostic(
        diagnostic_callback,
        "expert_pretrain_started",
        samples=EXPERT_PRETRAIN_SAMPLES,
        epochs=EXPERT_PRETRAIN_EPOCHS,
    )
    env = ContinuousNavigationEnv(
        spec=spec,
        max_steps=1000,
        randomize_start=True,
    )
    observations_0: list[np.ndarray] = []
    observations_1: list[np.ndarray] = []
    targets: list[list[float]] = []
    rollout_seed = 10_000
    while len(targets) < EXPERT_PRETRAIN_SAMPLES:
        for obs, target in _expert_rollout_samples(env=env, seed=rollout_seed):
            observations_0.append(obs["obs_0"].astype(np.uint8, copy=False))
            observations_1.append(obs["obs_1"].astype(np.float32, copy=False))
            targets.append(target)
            if len(targets) >= EXPERT_PRETRAIN_SAMPLES:
                break
        rollout_seed += 1
    env.close()
    _emit_training_diagnostic(
        diagnostic_callback,
        "expert_pretrain_samples_collected",
        samples=len(targets),
        image_dtype="uint8",
    )

    obs_0 = np.asarray(observations_0, dtype=np.uint8)
    obs_1 = np.asarray(observations_1, dtype=np.float32)
    target_actions = torch.as_tensor(np.asarray(targets), dtype=torch.float32)
    trainable_parameters = [
        *model.policy.features_extractor.parameters(),
        *model.policy.mlp_extractor.actor.parameters(),
        *model.policy.action_net.parameters(),
    ]
    optimizer = torch.optim.Adam(
        trainable_parameters,
        lr=EXPERT_PRETRAIN_LEARNING_RATE,
    )
    for epoch in range(EXPERT_PRETRAIN_EPOCHS):
        permutation = torch.randperm(EXPERT_PRETRAIN_SAMPLES)
        epoch_loss = 0.0
        for start in range(0, EXPERT_PRETRAIN_SAMPLES, EXPERT_PRETRAIN_BATCH_SIZE):
            indexes = permutation[start : start + EXPERT_PRETRAIN_BATCH_SIZE]
            numpy_indexes = indexes.numpy()
            features = model.policy.extract_features(
                {
                    "obs_0": torch.as_tensor(
                        obs_0[numpy_indexes],
                        dtype=torch.float32,
                    ),
                    "obs_1": torch.as_tensor(
                        obs_1[numpy_indexes],
                        dtype=torch.float32,
                    ),
                },
            )
            latent = model.policy.mlp_extractor.forward_actor(features)
            predicted_actions = model.policy.action_net(latent)
            loss = torch.nn.functional.mse_loss(
                predicted_actions,
                target_actions[indexes],
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach()) * len(indexes)
        _emit_training_diagnostic(
            diagnostic_callback,
            "expert_pretrain_epoch_finished",
            epoch=epoch + 1,
            loss=epoch_loss / EXPERT_PRETRAIN_SAMPLES,
        )
    _emit_training_diagnostic(diagnostic_callback, "expert_pretrain_finished")

"""Gymnasium-compatible grid-world environment for PPO training."""

from __future__ import annotations

from enum import IntEnum
from typing import ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from embodiedlab.training.training_models import GridPosition, GridWorldSpec


class Action(IntEnum):
    """Discrete movement actions available to the agent."""

    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


ACTION_TO_DELTA = {
    Action.UP: (0, -1),
    Action.RIGHT: (1, 0),
    Action.DOWN: (0, 1),
    Action.LEFT: (-1, 0),
}


class GridWorldTrainingEnv(gym.Env):
    """Grid-world environment with obstacle avoidance and goal-reaching reward."""

    metadata: ClassVar[dict] = {"render_modes": []}

    def __init__(self, spec: GridWorldSpec, max_steps: int = 100) -> None:
        """Initialise observation/action spaces and reset agent position."""
        super().__init__()
        self.spec = spec
        self.max_steps = max_steps

        self.action_space = spaces.Discrete(len(Action))
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(
                    low=np.array([0, 0], dtype=np.int32),
                    high=np.array([spec.width - 1, spec.height - 1], dtype=np.int32),
                    shape=(2,),
                    dtype=np.int32,
                ),
                "goal": spaces.Box(
                    low=np.array([0, 0], dtype=np.int32),
                    high=np.array([spec.width - 1, spec.height - 1], dtype=np.int32),
                    shape=(2,),
                    dtype=np.int32,
                ),
            },
        )

        self.agent_pos = np.array(
            [spec.robot_start.x, spec.robot_start.y],
            dtype=np.int32,
        )
        self.goal_pos = np.array([spec.goal.x, spec.goal.y], dtype=np.int32)
        self.steps = 0

    def _get_obs(self) -> dict:
        return {
            "agent": self.agent_pos.copy(),
            "goal": self.goal_pos.copy(),
        }

    def _get_info(self) -> dict:
        return {
            "distance": int(np.abs(self.agent_pos - self.goal_pos).sum()),
        }

    def reset(  # type: ignore[override]
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,  # noqa: ARG002
    ) -> tuple[dict, dict]:
        """Reset the environment to the initial state."""
        super().reset(seed=seed)
        self.agent_pos = np.array(
            [self.spec.robot_start.x, self.spec.robot_start.y],
            dtype=np.int32,
        )
        self.goal_pos = np.array([self.spec.goal.x, self.spec.goal.y], dtype=np.int32)
        self.steps = 0
        return self._get_obs(), self._get_info()

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:  # type: ignore[override]
        """Apply an action and return the next observation, reward, and flags."""
        prev_distance = int(np.abs(self.agent_pos - self.goal_pos).sum())

        self.steps += 1

        action = Action(action)
        dx, dy = ACTION_TO_DELTA[action]

        next_x = int(np.clip(self.agent_pos[0] + dx, 0, self.spec.width - 1))
        next_y = int(np.clip(self.agent_pos[1] + dy, 0, self.spec.height - 1))
        next_pos = GridPosition(x=next_x, y=next_y)

        if next_pos not in self.spec.obstacles:
            self.agent_pos = np.array([next_x, next_y], dtype=np.int32)

        terminated = bool(np.array_equal(self.agent_pos, self.goal_pos))
        truncated = self.steps >= self.max_steps

        new_distance = int(np.abs(self.agent_pos - self.goal_pos).sum())
        distance_delta = prev_distance - new_distance
        reward = 10.0 if terminated else (-0.2 + 0.5 * distance_delta)

        return self._get_obs(), reward, terminated, truncated, self._get_info()

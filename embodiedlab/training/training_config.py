"""Resolved PPO runtime configuration derived from a Scenario Bundle."""

from __future__ import annotations

from embodiedlab.schemas import TrainingSpec


class TrainingConfig(TrainingSpec):
    """Runtime view of the public training contract."""

    @property
    def max_steps(self) -> int:
        """Expose the environment's internal name for max episode steps."""
        return self.max_episode_steps

"""PPO training loops and post-training evaluation for navigation environments."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from pathlib import Path
from statistics import mean
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv

from embodiedlab.continuous_navigation_env import ContinuousNavigationEnv
from embodiedlab.training.navigation_final_expert import (
    pretrain_navigation_final_policy,
)
from embodiedlab.training.navigation_final_policy import (
    NavigationFinalPolicy,
    navigation_final_deterministic_raw_action,
)
from embodiedlab.training.training_config import (
    TrainingAlgorithm,
    TrainingConfig,
)

if TYPE_CHECKING:
    from embodiedlab.training.training_models import ContinuousNavigationSpec

TrainingProgressCallback = Callable[[int, int], None]
TrainingDiagnosticCallback = Callable[[str, dict[str, object]], None]
PROGRESS_LOG_INTERVAL_STEPS = 10_000
SUBPROC_START_METHOD = "fork"
TrainingEnv = ContinuousNavigationEnv | VecEnv


class TrainingProgressReporter(BaseCallback):
    """Report SB3 training progress at a fixed step interval."""

    def __init__(
        self,
        *,
        total_steps: int,
        progress_callback: TrainingProgressCallback,
        interval_steps: int = PROGRESS_LOG_INTERVAL_STEPS,
        diagnostic_callback: TrainingDiagnosticCallback | None = None,
    ) -> None:
        """Initialize the reporter with the total training budget."""
        super().__init__()
        self._total_steps = total_steps
        self._progress_callback = progress_callback
        self._interval_steps = interval_steps
        self._diagnostic_callback = diagnostic_callback
        self._last_reported_step = 0
        self._reported_first_rollout = False
        self._reported_first_step = False

    def _emit_diagnostic(self, event: str, **fields: object) -> None:
        if self._diagnostic_callback is None:
            return
        self._diagnostic_callback(
            event,
            {
                "current_step": int(self.num_timesteps),
                "total_steps": self._total_steps,
                **fields,
            },
        )

    def _on_training_start(self) -> None:
        self._emit_diagnostic("sb3_training_started")

    def _on_rollout_start(self) -> None:
        if self._reported_first_rollout:
            return
        self._reported_first_rollout = True
        self._emit_diagnostic("sb3_first_rollout_started")

    def _on_step(self) -> bool:
        current_step = int(self.num_timesteps)
        if not self._reported_first_step:
            self._reported_first_step = True
            self._emit_diagnostic("sb3_first_step")
        if current_step - self._last_reported_step >= self._interval_steps:
            self._last_reported_step = current_step
            self._progress_callback(current_step, self._total_steps)
        return True


def _termination_reason(
    *,
    terminated: bool,
    truncated: bool,
    collision: bool = False,
) -> str | None:
    if terminated and collision:
        return "collision"
    if terminated:
        return "goal_reached"
    if truncated:
        return "max_steps"
    return None


def build_continuous_replay_step(  # noqa: PLR0913
    *,
    episode_index: int,
    step_index: int,
    action: np.ndarray,
    obs: dict[str, np.ndarray],
    reward: float,
    info: dict[str, Any],
    terminated: bool,
    truncated: bool,
) -> dict[str, Any]:
    """Build a replay row directly from the continuous EnvForge runtime."""
    _ = obs
    action_array = np.asarray(action, dtype=np.float32)
    action_values = np.array(
        [
            info.get("applied_forward", action_array[0]),
            info.get("applied_turn", action_array[1]),
        ],
        dtype=np.float32,
    )
    reward_components = [
        {
            "name": str(component["name"]),
            "value": float(component["value"]),
        }
        for component in info.get("reward_components", [])
    ]
    if not reward_components:
        reward_components = [
            {
                "name": "reward",
                "value": float(reward),
            },
        ]

    events = []
    collision_id = info.get("collision_id")
    if collision_id is not None:
        events.append(
            {
                "type": "collision",
                "object_id": collision_id,
                "message": "Continuous movement was blocked",
            },
        )
    if terminated and not bool(info.get("collision")):
        events.append(
            {
                "type": "goal_reached",
                "object_id": "goal_001",
                "message": "Goal reached",
            },
        )

    return {
        "episode_id": f"episode_{episode_index + 1:04d}",
        "step_index": step_index,
        "time_seconds": round(step_index * 0.1, 6),
        "robot": {
            "position": {
                "x": float(info["robot_x"]),
                "z": float(info["robot_z"]),
            },
            "rotation_y_degrees": float(info["robot_rotation_y_degrees"]),
        },
        "action": {
            "values": [
                {
                    "name": "forward",
                    "value": float(action_values[0]),
                },
                {
                    "name": "turn",
                    "value": float(action_values[1]),
                },
            ],
        },
        "reward": {
            "total": reward,
            "components": reward_components,
        },
        "events": events,
        "sensors": [
            {
                "id": "front_distance",
                "type": "envforge_distance_sensor_meters",
                "value": float(info["front_distance"]),
            },
        ],
        "terminated": terminated or truncated,
        "termination_reason": _termination_reason(
            terminated=terminated,
            truncated=truncated,
            collision=bool(info.get("collision")),
        ),
    }


def evaluate_continuous_policy(
    model: PPO,
    env: ContinuousNavigationEnv,
    training: TrainingConfig,
) -> dict:
    """Run deterministic continuous-navigation rollouts and return statistics."""
    rewards: list[float] = []
    steps: list[int] = []
    successes = 0
    replay_steps: list[dict[str, Any]] = []

    for episode_index in range(training.eval_episodes):
        obs, _info = env.reset(seed=training.seed + episode_index)
        done = False
        episode_reward = 0.0
        episode_steps = 0
        terminated = False
        truncated = False

        while not done:
            action_array = _predict_navigation_final_raw_action(model, obs)
            obs, reward, terminated, truncated, _info = env.step(action_array)
            episode_reward += reward
            episode_steps += 1
            done = terminated or truncated
            if episode_index == 0:
                replay_steps.append(
                    build_continuous_replay_step(
                        episode_index=episode_index,
                        step_index=episode_steps - 1,
                        action=action_array,
                        obs=obs,
                        reward=reward,
                        info=_info,
                        terminated=terminated,
                        truncated=truncated,
                    ),
                )

        if terminated and not bool(_info.get("collision")):
            successes += 1

        rewards.append(episode_reward)
        steps.append(episode_steps)

    return {
        "episodes": training.eval_episodes,
        "success_rate": successes / training.eval_episodes,
        "avg_reward": mean(rewards),
        "avg_steps": mean(steps),
        "replay_steps": replay_steps,
    }


def _predict_navigation_final_raw_action(
    model: PPO,
    obs: dict[str, np.ndarray],
) -> np.ndarray:
    with torch.no_grad():
        action = navigation_final_deterministic_raw_action(
            model.policy,
            {
                "obs_0": torch.as_tensor(obs["obs_0"][None], dtype=torch.float32),
                "obs_1": torch.as_tensor(obs["obs_1"][None], dtype=torch.float32),
            },
        )
    return action.detach().cpu().numpy()[0].astype(np.float32)


def _emit_training_diagnostic(
    diagnostic_callback: TrainingDiagnosticCallback | None,
    event: str,
    **fields: object,
) -> None:
    if diagnostic_callback is None:
        return
    diagnostic_callback(event, fields)


def _configure_torch_threads(
    training: TrainingConfig,
    diagnostic_callback: TrainingDiagnosticCallback | None,
) -> None:
    if training.torch_num_threads is not None:
        torch.set_num_threads(training.torch_num_threads)
    _emit_training_diagnostic(
        diagnostic_callback,
        "torch_threads_configured",
        requested_torch_num_threads=training.torch_num_threads,
        torch_num_threads=torch.get_num_threads(),
    )


def _train_model(
    *,
    env: TrainingEnv,
    training: TrainingConfig,
    spec: ContinuousNavigationSpec | None = None,
    progress_callback: TrainingProgressCallback | None = None,
    diagnostic_callback: TrainingDiagnosticCallback | None = None,
) -> PPO:
    if training.algorithm != TrainingAlgorithm.PPO:
        msg = f"Unsupported algorithm: {training.algorithm}"
        raise ValueError(msg)

    _emit_training_diagnostic(
        diagnostic_callback,
        "ppo_model_construction_started",
        n_envs=training.n_envs,
        env_kind=_training_env_kind(training),
        torch_num_threads=torch.get_num_threads(),
        n_steps=training.n_steps,
        batch_size=training.batch_size,
        n_epochs=training.n_epochs,
        timesteps=training.timesteps,
    )
    model = PPO(
        policy=NavigationFinalPolicy,
        env=env,
        verbose=0,
        n_steps=training.n_steps,
        batch_size=training.batch_size,
        n_epochs=training.n_epochs,
        gamma=training.gamma,
        learning_rate=training.learning_rate,
        ent_coef=training.ent_coef,
        seed=training.seed,
    )
    _emit_training_diagnostic(
        diagnostic_callback,
        "ppo_model_constructed",
        n_envs=training.n_envs,
        torch_num_threads=torch.get_num_threads(),
    )
    if spec is not None:
        pretrain_navigation_final_policy(
            model=model,
            spec=spec,
            diagnostic_callback=diagnostic_callback,
        )
    callback = None
    if progress_callback is not None:
        callback = TrainingProgressReporter(
            total_steps=training.timesteps,
            progress_callback=progress_callback,
            diagnostic_callback=diagnostic_callback,
        )

    _emit_training_diagnostic(
        diagnostic_callback,
        "ppo_learn_started",
        total_steps=training.timesteps,
    )
    model.learn(total_timesteps=training.timesteps, callback=callback)
    _emit_training_diagnostic(
        diagnostic_callback,
        "ppo_learn_finished",
        total_steps=training.timesteps,
    )
    return model


def _build_training_env(
    spec: ContinuousNavigationSpec,
    training: TrainingConfig,
    diagnostic_callback: TrainingDiagnosticCallback | None = None,
) -> TrainingEnv:
    _emit_training_diagnostic(
        diagnostic_callback,
        "training_env_build_started",
        n_envs=training.n_envs,
        env_kind=_training_env_kind(training),
        max_steps=training.max_steps,
        obstacle_count=len(spec.obstacles),
    )
    if training.n_envs == 1:
        env = ContinuousNavigationEnv(
            spec=spec,
            max_steps=training.max_steps,
            randomize_start=True,
        )
        _emit_training_diagnostic(
            diagnostic_callback,
            "training_env_built",
            env_kind="single",
            n_envs=training.n_envs,
        )
        return env

    env_fns = [
        partial(_make_continuous_navigation_env, spec, training.max_steps)
        for _ in range(training.n_envs)
    ]
    env = SubprocVecEnv(env_fns, start_method=SUBPROC_START_METHOD)
    _emit_training_diagnostic(
        diagnostic_callback,
        "training_env_built",
        env_kind="subproc_vec",
        n_envs=training.n_envs,
        start_method=SUBPROC_START_METHOD,
    )
    return env


def _training_env_kind(training: TrainingConfig) -> str:
    return "single" if training.n_envs == 1 else "subproc_vec"


def _make_continuous_navigation_env(
    spec: ContinuousNavigationSpec,
    max_steps: int,
) -> ContinuousNavigationEnv:
    return ContinuousNavigationEnv(
        spec=spec,
        max_steps=max_steps,
        randomize_start=True,
    )


def _save_model(model: PPO, model_output_path: str | None) -> None:
    if model_output_path is None:
        return

    output_path = Path(model_output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path))


def run_continuous_navigation_training(
    spec: ContinuousNavigationSpec,
    training: TrainingConfig,
    model_output_path: str | None = None,
    progress_callback: TrainingProgressCallback | None = None,
    diagnostic_callback: TrainingDiagnosticCallback | None = None,
) -> dict:
    """Train a PPO policy on the continuous navigation spec and evaluate it."""
    _emit_training_diagnostic(
        diagnostic_callback,
        "continuous_training_started",
        n_envs=training.n_envs,
        env_kind=_training_env_kind(training),
        cpu_count=training.cpu_count,
        requested_torch_num_threads=training.torch_num_threads,
        timesteps=training.timesteps,
    )
    _configure_torch_threads(training, diagnostic_callback)
    train_env = _build_training_env(
        spec=spec,
        training=training,
        diagnostic_callback=diagnostic_callback,
    )
    _emit_training_diagnostic(diagnostic_callback, "eval_env_build_started")
    eval_env = ContinuousNavigationEnv(
        spec=spec,
        max_steps=training.max_steps,
        randomize_start=True,
    )
    _emit_training_diagnostic(diagnostic_callback, "eval_env_built")

    try:
        model = _train_model(
            env=train_env,
            training=training,
            spec=spec,
            progress_callback=progress_callback,
            diagnostic_callback=diagnostic_callback,
        )
        _emit_training_diagnostic(diagnostic_callback, "evaluation_started")
        evaluation = evaluate_continuous_policy(
            model=model,
            env=eval_env,
            training=training,
        )
        _emit_training_diagnostic(
            diagnostic_callback,
            "evaluation_finished",
            success_rate=evaluation["success_rate"],
            avg_steps=evaluation["avg_steps"],
        )
        _emit_training_diagnostic(diagnostic_callback, "model_save_started")
        _save_model(model=model, model_output_path=model_output_path)
        _emit_training_diagnostic(diagnostic_callback, "model_saved")

        return {
            "policy": training.algorithm.value,
            "runtime": "continuous_navigation",
            "score": evaluation["avg_reward"],
            "episodes": evaluation["episodes"],
            "obstacle_count": len(spec.obstacles),
            "goal": {
                "x": spec.goal.x,
                "z": spec.goal.z,
                "radius": spec.goal.radius,
            },
            "robot_start": {
                "x": spec.robot_start.x,
                "z": spec.robot_start.z,
                "rotation_y_degrees": spec.robot_start.rotation_y_degrees,
            },
            "robot_type": spec.robot_type,
            "success_rate": evaluation["success_rate"],
            "avg_reward": evaluation["avg_reward"],
            "avg_steps": evaluation["avg_steps"],
            "training_timesteps": training.timesteps,
            "training_seed": training.seed,
            "replay_steps": evaluation["replay_steps"],
        }

    finally:
        train_env.close()
        eval_env.close()

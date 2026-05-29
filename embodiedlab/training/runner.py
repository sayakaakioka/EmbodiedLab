"""PPO training loops and post-training evaluation for navigation environments."""

from __future__ import annotations

from pathlib import Path
from statistics import mean
from typing import TYPE_CHECKING, Any

import numpy as np
from stable_baselines3 import PPO

from embodiedlab.continuous_navigation_env import ContinuousNavigationEnv
from embodiedlab.training.training_config import TrainingAlgorithm, TrainingConfig

if TYPE_CHECKING:
    from embodiedlab.training.training_models import ContinuousNavigationSpec


def _termination_reason(*, terminated: bool, truncated: bool) -> str | None:
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
    action_values = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
    reward_components = [
        {
            "name": "step_penalty",
            "value": -0.01,
        },
    ]
    distance_delta = float(info.get("distance_delta", 0.0))
    if distance_delta:
        reward_components.append(
            {
                "name": "goal_progress",
                "value": 0.5 * distance_delta,
            },
        )
    if info.get("collision"):
        reward_components.append(
            {
                "name": "collision_penalty",
                "value": -5.0,
            },
        )
    if terminated:
        reward_components.append(
            {
                "name": "goal_reached",
                "value": 10.0,
            },
        )

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
    if terminated:
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
                "x": float(obs["robot"][0]),
                "z": float(obs["robot"][1]),
            },
            "rotation_y_degrees": float(obs["robot"][2]),
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
            action, _states = model.predict(obs, deterministic=True)
            action_array = np.asarray(action, dtype=np.float32)
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

        if terminated:
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


def _train_model(
    *,
    env: ContinuousNavigationEnv,
    training: TrainingConfig,
) -> PPO:
    if training.algorithm != TrainingAlgorithm.PPO:
        msg = f"Unsupported algorithm: {training.algorithm}"
        raise ValueError(msg)

    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        verbose=0,
        n_steps=training.n_steps,
        batch_size=training.batch_size,
        gamma=training.gamma,
        learning_rate=training.learning_rate,
        ent_coef=training.ent_coef,
        seed=training.seed,
    )
    model.learn(total_timesteps=training.timesteps)
    return model


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
) -> dict:
    """Train a PPO policy on the continuous navigation spec and evaluate it."""
    train_env = ContinuousNavigationEnv(spec=spec, max_steps=training.max_steps)
    eval_env = ContinuousNavigationEnv(spec=spec, max_steps=training.max_steps)

    try:
        model = _train_model(env=train_env, training=training)
        evaluation = evaluate_continuous_policy(
            model=model,
            env=eval_env,
            training=training,
        )
        _save_model(model=model, model_output_path=model_output_path)

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

"""PPO training loop and post-training evaluation for grid-world environments."""

from __future__ import annotations

from pathlib import Path
from statistics import mean
from typing import TYPE_CHECKING, Any

from stable_baselines3 import PPO

from embodiedlab.gridworld_env import Action, GridWorldTrainingEnv
from embodiedlab.training.training_config import TrainingAlgorithm, TrainingConfig

if TYPE_CHECKING:
    from embodiedlab.training.training_models import GridWorldSpec

ACTION_REPLAY_VALUES = {
    Action.UP: {"forward": 1.0, "turn": 0.0, "rotation_y_degrees": 0.0},
    Action.RIGHT: {"forward": 0.0, "turn": 1.0, "rotation_y_degrees": 90.0},
    Action.DOWN: {"forward": -1.0, "turn": 0.0, "rotation_y_degrees": 180.0},
    Action.LEFT: {"forward": 0.0, "turn": -1.0, "rotation_y_degrees": 270.0},
}


def build_grid_replay_step(  # noqa: PLR0913
    *,
    episode_index: int,
    step_index: int,
    action: int,
    obs: dict[str, Any],
    reward: float,
    info: dict[str, Any],
    terminated: bool,
    truncated: bool,
    envforge_origin_x: float = 0.0,
    envforge_origin_z: float = 0.0,
    cell_size_meters: float = 1.0,
) -> dict[str, Any]:
    """Build a JsonUtility-friendly replay row from the temporary grid runtime."""
    action_values = ACTION_REPLAY_VALUES[Action(action)]
    reward_components = [
        {
            "name": "step_penalty",
            "value": -0.2,
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
    if terminated:
        reward_components.append(
            {
                "name": "goal_reached",
                "value": 10.0,
            },
        )

    events = []
    if info.get("blocked"):
        events.append(
            {
                "type": "collision",
                "object_id": None,
                "message": "Grid movement was blocked",
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
                "x": envforge_origin_x + float(obs["agent"][0]) * cell_size_meters,
                "z": envforge_origin_z + float(obs["agent"][1]) * cell_size_meters,
            },
            "rotation_y_degrees": action_values["rotation_y_degrees"],
        },
        "action": {
            "values": [
                {
                    "name": "forward",
                    "value": action_values["forward"],
                },
                {
                    "name": "turn",
                    "value": action_values["turn"],
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
                "type": "envforge_manhattan_distance_meters",
                "value": float(info["distance"]) * cell_size_meters,
            },
        ],
        "terminated": terminated or truncated,
        "termination_reason": (
            "goal_reached" if terminated else "max_steps" if truncated else None
        ),
    }


def evaluate_policy(
    model: PPO,
    env: GridWorldTrainingEnv,
    training: TrainingConfig,
) -> dict:
    """Run deterministic rollouts and return episode statistics."""
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
            obs, reward, terminated, truncated, _info = env.step(int(action))
            episode_reward += reward
            episode_steps += 1
            done = terminated or truncated
            if episode_index == 0:
                replay_steps.append(
                    build_grid_replay_step(
                        episode_index=episode_index,
                        step_index=episode_steps - 1,
                        action=int(action),
                        obs=obs,
                        reward=reward,
                        info=_info,
                        terminated=terminated,
                        truncated=truncated,
                        envforge_origin_x=env.spec.envforge_origin_x,
                        envforge_origin_z=env.spec.envforge_origin_z,
                        cell_size_meters=env.spec.cell_size_meters,
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


def run_gridworld_training(
    spec: GridWorldSpec,
    training: TrainingConfig,
    model_output_path: str | None = None,
) -> dict:
    """Train a PPO policy on the given spec, evaluate it, and optionally save it."""
    if training.algorithm != TrainingAlgorithm.PPO:
        msg = f"Unsupported algorithm: {training.algorithm}"
        raise ValueError(msg)

    train_env = GridWorldTrainingEnv(spec=spec, max_steps=training.max_steps)
    eval_env = GridWorldTrainingEnv(spec=spec, max_steps=training.max_steps)

    try:
        model = PPO(
            policy="MultiInputPolicy",
            env=train_env,
            verbose=0,
            n_steps=training.n_steps,
            batch_size=training.batch_size,
            gamma=training.gamma,
            learning_rate=training.learning_rate,
            ent_coef=training.ent_coef,
            seed=training.seed,
        )
        model.learn(total_timesteps=training.timesteps)
        evaluation = evaluate_policy(model=model, env=eval_env, training=training)

        if model_output_path is not None:
            output_path = Path(model_output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            model.save(str(output_path))

        return {
            "policy": training.algorithm.value,
            "score": evaluation["avg_reward"],
            "grid_width": spec.width,
            "grid_height": spec.height,
            "episodes": evaluation["episodes"],
            "obstacle_count": len(spec.obstacles),
            "goal": {
                "x": spec.goal.x,
                "y": spec.goal.y,
            },
            "robot_start": {
                "x": spec.robot_start.x,
                "y": spec.robot_start.y,
            },
            "robot_type": spec.robot_type,
            "success_rate": evaluation["success_rate"],
            "avg_reward": evaluation["avg_reward"],
            "avg_steps": evaluation["avg_steps"],
            "training_timesteps": training.timesteps,
            "training_seed": training.seed,
        }

    finally:
        train_env.close()
        eval_env.close()

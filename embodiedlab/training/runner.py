"""PPO training loops and post-training evaluation for navigation environments."""

from __future__ import annotations

import os
from collections.abc import Callable
from functools import partial
from pathlib import Path
from statistics import mean
from typing import TYPE_CHECKING, Any

import numpy as np
import stable_baselines3
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv

from embodiedlab.continuous_navigation_env import ContinuousNavigationEnv
from embodiedlab.schemas import MAX_TRAINING_CPU_COUNT, TrainingAlgorithm
from embodiedlab.training.navigation_final_policy import (
    NavigationFinalPolicy,
    navigation_final_deterministic_raw_action,
)
from embodiedlab.training.replay_bundle import ReplayBundleWriter

if TYPE_CHECKING:
    from embodiedlab.schemas import TrainingSpec
    from embodiedlab.training.training_models import ContinuousNavigationSpec

TrainingProgressCallback = Callable[[int, int], None]
TrainingDiagnosticCallback = Callable[[str, dict[str, object]], None]
PROGRESS_LOG_INTERVAL_STEPS = 10_000
SUBPROC_START_METHOD = "forkserver"
TrainingEnv = ContinuousNavigationEnv | VecEnv


class TrainingProgressReporter(BaseCallback):
    """Report progress and record replay timeline data during SB3 training."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        total_steps: int,
        progress_callback: TrainingProgressCallback | None,
        interval_steps: int = PROGRESS_LOG_INTERVAL_STEPS,
        diagnostic_callback: TrainingDiagnosticCallback | None = None,
        replay_writer: ReplayBundleWriter | None = None,
        training: TrainingSpec | None = None,
        eval_spec: ContinuousNavigationSpec | None = None,
    ) -> None:
        """Initialize the reporter with progress and replay recording settings."""
        super().__init__()
        self._total_steps = total_steps
        self._progress_callback = progress_callback
        self._interval_steps = interval_steps
        self._diagnostic_callback = diagnostic_callback
        self._replay_writer = replay_writer
        self._training = training
        self._eval_spec = eval_spec
        self._last_reported_step = 0
        self._next_eval_step = 0
        self._reported_first_rollout = False
        self._reported_first_step = False
        self._episode_indices: list[int] = []
        self._episode_steps: list[int] = []

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
        env_count = max(1, self.training_env.num_envs)
        self._episode_indices = [0 for _ in range(env_count)]
        self._episode_steps = [0 for _ in range(env_count)]
        self._next_eval_step = (
            self._training.replay_eval_interval_steps
            if self._training is not None
            else self._total_steps
        )
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
        self._record_training_replay(current_step)
        if current_step - self._last_reported_step >= self._interval_steps:
            self._last_reported_step = current_step
            if self._progress_callback is not None:
                self._progress_callback(current_step, self._total_steps)
        self._record_checkpoint_eval_if_needed(current_step)
        return True

    def _record_training_replay(self, current_step: int) -> None:
        if self._replay_writer is None:
            return
        actions = np.asarray(self.locals.get("actions", []), dtype=np.float32)
        rewards = np.asarray(self.locals.get("rewards", []), dtype=np.float32)
        dones = np.asarray(self.locals.get("dones", []), dtype=bool)
        infos = list(self.locals.get("infos", []))
        if len(infos) == 0:
            return
        for env_index, info in enumerate(infos):
            if env_index >= len(self._episode_indices):
                continue
            action = actions[env_index] if actions.ndim > 1 else actions
            reward = float(rewards[env_index]) if rewards.size > env_index else 0.0
            done = bool(dones[env_index]) if dones.size > env_index else False
            step = build_continuous_replay_step(
                goal_id=self._eval_spec.goal.goal_id,
                distance_sensor_id=self._eval_spec.distance_sensor_id,
                step_duration_seconds=self._eval_spec.step_duration_seconds,
                episode_index=self._episode_indices[env_index],
                step_index=self._episode_steps[env_index],
                action=action,
                obs={},
                reward=reward,
                info=info,
                terminated=done and not bool(info.get("TimeLimit.truncated", False)),
                truncated=done and bool(info.get("TimeLimit.truncated", False)),
                phase="train",
                checkpoint_step=current_step,
                env_index=env_index,
                policy_mode="stochastic",
            )
            self._replay_writer.record_train_step(step)
            self._episode_steps[env_index] += 1
            if done:
                self._episode_indices[env_index] += 1
                self._episode_steps[env_index] = 0

    def _record_checkpoint_eval_if_needed(self, current_step: int) -> None:
        if (
            self._replay_writer is None
            or self._training is None
            or self._eval_spec is None
            or self._training.replay_eval_interval_steps <= 0
            or current_step < self._next_eval_step
            or current_step >= self._total_steps
        ):
            return
        eval_env = ContinuousNavigationEnv(
            spec=self._eval_spec,
            max_episode_steps=self._training.max_episode_steps,
            randomize_start=self._training.randomize_start,
        )
        try:
            evaluation = evaluate_continuous_policy(
                model=self.model,
                env=eval_env,
                training=self._training,
                phase="eval",
                checkpoint_step=current_step,
            )
        finally:
            eval_env.close()
        self._replay_writer.write_eval_checkpoint(
            checkpoint_step=current_step,
            steps=evaluation["replay_steps"],
            success_rate=evaluation["success_rate"],
            avg_reward=evaluation["avg_reward"],
            avg_steps=evaluation["avg_steps"],
        )
        self._emit_diagnostic(
            "checkpoint_evaluation_finished",
            checkpoint_step=current_step,
            success_rate=evaluation["success_rate"],
            avg_steps=evaluation["avg_steps"],
        )
        self._next_eval_step += self._training.replay_eval_interval_steps


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
    goal_id: str,
    distance_sensor_id: str | None,
    step_duration_seconds: float,
    episode_index: int,
    step_index: int,
    action: np.ndarray,
    obs: dict[str, np.ndarray],
    reward: float,
    info: dict[str, Any],
    terminated: bool,
    truncated: bool,
    phase: str = "eval",
    checkpoint_step: int = 0,
    env_index: int = 0,
    policy_mode: str = "deterministic",
) -> dict[str, Any]:
    """Build a replay row directly from the continuous runtime."""
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
                "object_id": goal_id,
                "message": "Goal reached",
            },
        )

    sensors = [
        {
            "id": "camera_mount_height",
            "type": "camera_mount_height_meters",
            "value": float(info["camera_mount_height_meters"]),
        },
    ]
    if distance_sensor_id is not None:
        sensors.insert(
            0,
            {
                "id": distance_sensor_id,
                "type": "distance_meters",
                "value": float(info["front_distance"]),
            },
        )

    return {
        "phase": phase,
        "checkpoint_step": checkpoint_step,
        "env_index": env_index,
        "policy_mode": policy_mode,
        "episode_id": f"{phase}_env_{env_index:02d}_episode_{episode_index + 1:06d}",
        "step_index": step_index,
        "time_seconds": round(step_index * step_duration_seconds, 6),
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
        "sensors": sensors,
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
    training: TrainingSpec,
    *,
    phase: str = "eval",
    checkpoint_step: int = 0,
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
            replay_steps.append(
                build_continuous_replay_step(
                    goal_id=env.spec.goal.goal_id,
                    distance_sensor_id=env.spec.distance_sensor_id,
                    step_duration_seconds=env.spec.step_duration_seconds,
                    episode_index=episode_index,
                    step_index=episode_steps - 1,
                    action=action_array,
                    obs=obs,
                    reward=reward,
                    info=_info,
                    terminated=terminated,
                    truncated=truncated,
                    phase=phase,
                    checkpoint_step=checkpoint_step,
                    env_index=0,
                    policy_mode="deterministic",
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
                name: torch.as_tensor(value[None], dtype=torch.float32)
                for name, value in obs.items()
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


def _configure_cpu_count(
    training: TrainingSpec,
    diagnostic_callback: TrainingDiagnosticCallback | None,
) -> int:
    available_cpus = min(
        MAX_TRAINING_CPU_COUNT,
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else (os.process_cpu_count() or 1),
    )
    requested_cpus = training.cpu_count
    effective_cpus = requested_cpus or available_cpus
    if effective_cpus > available_cpus:
        msg = (
            f"requested cpu_count {effective_cpus} exceeds available CPU count "
            f"{available_cpus}"
        )
        raise ValueError(msg)
    if requested_cpus is not None and requested_cpus < available_cpus:
        if not hasattr(os, "sched_setaffinity"):
            msg = "cpu_count overrides require process CPU affinity support"
            raise ValueError(msg)
        current_affinity = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, current_affinity[:requested_cpus])
        effective_cpus = len(os.sched_getaffinity(0))
    if training.n_envs > effective_cpus:
        msg = "n_envs must not exceed the effective CPU count"
        raise ValueError(msg)
    if (
        training.torch_num_threads is not None
        and training.torch_num_threads > effective_cpus
    ):
        msg = "torch_num_threads must not exceed the effective CPU count"
        raise ValueError(msg)
    _emit_training_diagnostic(
        diagnostic_callback,
        "cpu_count_configured",
        requested_cpu_count=requested_cpus,
        available_cpu_count=available_cpus,
        cpu_count=effective_cpus,
    )
    return effective_cpus


def _configure_torch_threads(
    training: TrainingSpec,
    effective_cpu_count: int,
    diagnostic_callback: TrainingDiagnosticCallback | None,
) -> int:
    resolved_threads = training.torch_num_threads or effective_cpu_count
    torch.set_num_threads(resolved_threads)
    effective_threads = torch.get_num_threads()
    if effective_threads > effective_cpu_count:
        msg = "effective torch thread count exceeds the effective CPU count"
        raise ValueError(msg)
    _emit_training_diagnostic(
        diagnostic_callback,
        "torch_threads_configured",
        requested_torch_num_threads=training.torch_num_threads,
        torch_num_threads=effective_threads,
    )
    return effective_threads


def _train_model(  # noqa: PLR0913
    *,
    env: TrainingEnv,
    training: TrainingSpec,
    progress_callback: TrainingProgressCallback | None = None,
    diagnostic_callback: TrainingDiagnosticCallback | None = None,
    replay_writer: ReplayBundleWriter | None = None,
    eval_spec: ContinuousNavigationSpec | None = None,
) -> PPO:
    if training.algorithm != TrainingAlgorithm.PPO:
        msg = f"Unsupported algorithm: {training.algorithm}"
        raise ValueError(msg)
    contract_spec = eval_spec or getattr(env, "spec", None)
    if contract_spec is None:
        msg = "training requires an explicit navigation contract"
        raise ValueError(msg)
    goal_values = contract_spec.goal_vector.values

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
        gae_lambda=training.gae_lambda,
        learning_rate=training.learning_rate,
        clip_range=training.clip_range,
        clip_range_vf=training.clip_range_vf,
        normalize_advantage=training.normalize_advantage,
        ent_coef=training.ent_coef,
        vf_coef=training.vf_coef,
        max_grad_norm=training.max_grad_norm,
        use_sde=training.use_sde,
        sde_sample_freq=training.sde_sample_freq,
        target_kl=training.target_kl,
        stats_window_size=training.stats_window_size,
        seed=training.seed,
        device=training.device.value,
        policy_kwargs={
            "features_extractor_kwargs": {
                "image_observation_name": contract_spec.camera.observation_name,
                "goal_vector_observation_name": (
                    contract_spec.goal_vector.observation_name
                ),
                "goal_angle_index": goal_values.index("goal_angle_degrees"),
                "goal_distance_index": goal_values.index("goal_distance_meters"),
            },
        },
    )
    _emit_training_diagnostic(
        diagnostic_callback,
        "ppo_model_constructed",
        n_envs=training.n_envs,
        torch_num_threads=torch.get_num_threads(),
    )
    callback = TrainingProgressReporter(
        total_steps=training.timesteps,
        progress_callback=progress_callback,
        diagnostic_callback=diagnostic_callback,
        replay_writer=replay_writer,
        training=training,
        eval_spec=eval_spec,
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
    training: TrainingSpec,
    diagnostic_callback: TrainingDiagnosticCallback | None = None,
) -> TrainingEnv:
    _emit_training_diagnostic(
        diagnostic_callback,
        "training_env_build_started",
        n_envs=training.n_envs,
        env_kind=_training_env_kind(training),
        max_episode_steps=training.max_episode_steps,
        obstacle_count=len(spec.obstacles),
    )
    if training.n_envs == 1:
        env = ContinuousNavigationEnv(
            spec=spec,
            max_episode_steps=training.max_episode_steps,
            randomize_start=training.randomize_start,
        )
        _emit_training_diagnostic(
            diagnostic_callback,
            "training_env_built",
            env_kind="single",
            n_envs=training.n_envs,
        )
        return env

    env_fns = [
        partial(
            _make_continuous_navigation_env,
            spec,
            training.max_episode_steps,
            randomize_start=training.randomize_start,
        )
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


def _training_env_kind(training: TrainingSpec) -> str:
    return "single" if training.n_envs == 1 else "subproc_vec"


def _make_continuous_navigation_env(
    spec: ContinuousNavigationSpec,
    max_episode_steps: int,
    *,
    randomize_start: bool,
) -> ContinuousNavigationEnv:
    return ContinuousNavigationEnv(
        spec=spec,
        max_episode_steps=max_episode_steps,
        randomize_start=randomize_start,
    )


def _save_model(model: PPO, model_output_path: str | None) -> None:
    if model_output_path is None:
        return

    output_path = Path(model_output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path))


def run_continuous_navigation_training(  # noqa: PLR0913
    spec: ContinuousNavigationSpec,
    training: TrainingSpec,
    model_output_path: str | None = None,
    progress_callback: TrainingProgressCallback | None = None,
    diagnostic_callback: TrainingDiagnosticCallback | None = None,
    scenario_id: str = "unknown_scenario",
    job_id: str = "unknown_job",
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
    effective_cpu_count = _configure_cpu_count(training, diagnostic_callback)
    effective_torch_num_threads = _configure_torch_threads(
        training,
        effective_cpu_count,
        diagnostic_callback,
    )
    replay_root = (
        Path(model_output_path).parent / "replay_bundle"
        if model_output_path
        else Path.cwd() / "replay_bundle"
    )
    replay_writer = ReplayBundleWriter(
        root_dir=replay_root,
        job_id=job_id,
        scenario_id=scenario_id,
        total_timesteps=training.timesteps,
        train_chunk_steps=training.replay_train_chunk_steps,
    )
    train_env = _build_training_env(
        spec=spec,
        training=training,
        diagnostic_callback=diagnostic_callback,
    )
    _emit_training_diagnostic(diagnostic_callback, "eval_env_build_started")
    eval_env = ContinuousNavigationEnv(
        spec=spec,
        max_episode_steps=training.max_episode_steps,
        randomize_start=training.randomize_start,
    )
    _emit_training_diagnostic(diagnostic_callback, "eval_env_built")

    try:
        model = _train_model(
            env=train_env,
            training=training,
            progress_callback=progress_callback,
            diagnostic_callback=diagnostic_callback,
            replay_writer=replay_writer,
            eval_spec=spec,
        )
        _emit_training_diagnostic(diagnostic_callback, "evaluation_started")
        evaluation = evaluate_continuous_policy(
            model=model,
            env=eval_env,
            training=training,
            phase="eval",
            checkpoint_step=training.timesteps,
        )
        replay_writer.write_eval_checkpoint(
            checkpoint_step=training.timesteps,
            steps=evaluation["replay_steps"],
            success_rate=evaluation["success_rate"],
            avg_reward=evaluation["avg_reward"],
            avg_steps=evaluation["avg_steps"],
        )
        replay_writer.finish()
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
            "summary": {
                "success_rate": evaluation["success_rate"],
                "average_episode_reward": evaluation["avg_reward"],
                "average_episode_steps": evaluation["avg_steps"],
                "configuration": {
                    "library": "stable-baselines3",
                    "library_version": stable_baselines3.__version__,
                    "algorithm": training.algorithm.value,
                    "device": training.device.value,
                    "timesteps": training.timesteps,
                    "seed": training.seed,
                    "max_episode_steps": training.max_episode_steps,
                    "n_envs": training.n_envs,
                    "requested_cpu_count": training.cpu_count,
                    "cpu_count": effective_cpu_count,
                    "requested_torch_num_threads": training.torch_num_threads,
                    "torch_num_threads": effective_torch_num_threads,
                    "n_steps": training.n_steps,
                    "batch_size": training.batch_size,
                    "n_epochs": training.n_epochs,
                    "gamma": training.gamma,
                    "gae_lambda": training.gae_lambda,
                    "learning_rate": training.learning_rate,
                    "clip_range": training.clip_range,
                    "clip_range_vf": training.clip_range_vf,
                    "normalize_advantage": training.normalize_advantage,
                    "ent_coef": training.ent_coef,
                    "vf_coef": training.vf_coef,
                    "max_grad_norm": training.max_grad_norm,
                    "use_sde": training.use_sde,
                    "sde_sample_freq": training.sde_sample_freq,
                    "target_kl": training.target_kl,
                    "stats_window_size": training.stats_window_size,
                    "eval_episodes": training.eval_episodes,
                    "replay_eval_interval_steps": (training.replay_eval_interval_steps),
                    "replay_train_chunk_steps": training.replay_train_chunk_steps,
                    "randomize_start": training.randomize_start,
                },
            },
            "replay_bundle_dir": str(replay_writer.root_dir),
        }

    finally:
        train_env.close()
        eval_env.close()

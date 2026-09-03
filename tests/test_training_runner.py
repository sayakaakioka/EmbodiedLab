import numpy as np
import pytest

from embodiedlab.continuous_navigation_env import ContinuousNavigationEnv
from embodiedlab.training.runner import (
    TrainingProgressReporter,
    _build_training_env,
    _configure_cpu_count,
    _predict_navigation_final_raw_action,
    _train_model,
    build_continuous_replay_step,
    evaluate_continuous_policy,
)
from embodiedlab.training.training_converter import convert_scenario_to_spec
from tests.fakes import scenario_bundle, training_spec


def test_training_configures_torch_threads(monkeypatch):
    calls = []

    monkeypatch.setattr(
        "embodiedlab.training.runner.torch.set_num_threads",
        calls.append,
    )
    monkeypatch.setattr("embodiedlab.training.runner.torch.get_num_threads", lambda: 1)

    from embodiedlab.training.runner import _configure_torch_threads

    events = []

    _configure_torch_threads(
        training_spec(torch_num_threads=1),
        1,
        lambda event, fields: events.append((event, fields)),
    )

    assert calls == [1]
    assert events == [
        (
            "torch_threads_configured",
            {
                "requested_torch_num_threads": 1,
                "torch_num_threads": 1,
            },
        ),
    ]


def test_unspecified_torch_threads_resolve_to_effective_cpu_count(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "embodiedlab.training.runner.torch.set_num_threads",
        calls.append,
    )
    monkeypatch.setattr("embodiedlab.training.runner.torch.get_num_threads", lambda: 2)

    from embodiedlab.training.runner import _configure_torch_threads

    effective = _configure_torch_threads(
        training_spec(torch_num_threads=None),
        2,
        None,
    )

    assert calls == [2]
    assert effective == 2


def test_training_applies_requested_cpu_affinity(monkeypatch):
    affinity = {0, 1, 2, 3}

    monkeypatch.setattr(
        "embodiedlab.training.runner.os.sched_getaffinity",
        lambda _pid: affinity,
        raising=False,
    )

    def set_affinity(_pid, cpus):
        affinity.clear()
        affinity.update(cpus)

    monkeypatch.setattr(
        "embodiedlab.training.runner.os.sched_setaffinity",
        set_affinity,
        raising=False,
    )
    events = []

    effective = _configure_cpu_count(
        training_spec(cpu_count=2, n_envs=2, torch_num_threads=1),
        lambda event, fields: events.append((event, fields)),
    )

    assert effective == 2
    assert affinity == {0, 1}
    assert events == [
        (
            "cpu_count_configured",
            {
                "requested_cpu_count": 2,
                "available_cpu_count": 4,
                "cpu_count": 2,
            },
        ),
    ]


def test_training_rejects_requested_cpu_count_above_available(monkeypatch):
    monkeypatch.setattr(
        "embodiedlab.training.runner.os.sched_getaffinity",
        lambda _pid: {0, 1},
        raising=False,
    )

    with pytest.raises(ValueError, match="exceeds available CPU count"):
        _configure_cpu_count(training_spec(cpu_count=3), None)


def test_build_training_env_randomizes_start_for_single_env():
    scenario = scenario_bundle()
    spec = convert_scenario_to_spec(scenario)
    training = training_spec(n_envs=1)

    env = _build_training_env(spec=spec, training=training)

    assert env.randomize_start is True


def test_build_training_env_uses_subproc_vec_env_automatically_for_multiple_envs():
    scenario = scenario_bundle()
    spec = convert_scenario_to_spec(scenario)
    training = training_spec(n_envs=2)
    events = []

    env = _build_training_env(
        spec=spec,
        training=training,
        diagnostic_callback=lambda event, fields: events.append((event, fields)),
    )

    try:
        assert env.num_envs == 2
        assert events[-1] == (
            "training_env_built",
            {
                "env_kind": "subproc_vec",
                "n_envs": 2,
                "start_method": "forkserver",
            },
        )
    finally:
        env.close()


def test_navigation_final_raw_prediction_matches_sb3_deterministic_action():
    scenario = scenario_bundle()
    spec = convert_scenario_to_spec(scenario)
    env = ContinuousNavigationEnv(spec=spec, max_episode_steps=10)
    training = training_spec(timesteps=1, n_steps=8, batch_size=4, seed=10)
    model = _train_model(env=env, training=training)
    obs, _info = env.reset(seed=10)

    sb3_action, _state = model.predict(obs, deterministic=True)
    action = _predict_navigation_final_raw_action(model, obs)

    np.testing.assert_allclose(action, sb3_action)


def test_train_model_passes_declared_ppo_configuration(monkeypatch):
    scenario = scenario_bundle()
    spec = convert_scenario_to_spec(scenario)
    env = ContinuousNavigationEnv(spec=spec, max_episode_steps=10)
    training = training_spec(
        timesteps=1,
        n_steps=8,
        batch_size=4,
        n_epochs=3,
        gamma=0.91,
        gae_lambda=0.92,
        learning_rate=0.001,
        clip_range=0.15,
        clip_range_vf=0.3,
        normalize_advantage=False,
        ent_coef=0.02,
        vf_coef=0.7,
        max_grad_norm=0.8,
        use_sde=False,
        sde_sample_freq=-1,
        target_kl=0.04,
        stats_window_size=50,
    )
    captured = {}
    events = []

    class FakePPO:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def learn(self, total_timesteps, callback=None):
            captured["total_timesteps"] = total_timesteps
            captured["callback"] = callback

    monkeypatch.setattr("embodiedlab.training.runner.PPO", FakePPO)

    model = _train_model(
        env=env,
        training=training,
        diagnostic_callback=lambda event, fields: events.append((event, fields)),
    )

    assert model is not None
    assert {
        key: captured[key]
        for key in (
            "n_steps",
            "batch_size",
            "n_epochs",
            "gamma",
            "gae_lambda",
            "learning_rate",
            "clip_range",
            "clip_range_vf",
            "normalize_advantage",
            "ent_coef",
            "vf_coef",
            "max_grad_norm",
            "use_sde",
            "sde_sample_freq",
            "target_kl",
            "stats_window_size",
        )
    } == {
        "n_steps": 8,
        "batch_size": 4,
        "n_epochs": 3,
        "gamma": 0.91,
        "gae_lambda": 0.92,
        "learning_rate": 0.001,
        "clip_range": 0.15,
        "clip_range_vf": 0.3,
        "normalize_advantage": False,
        "ent_coef": 0.02,
        "vf_coef": 0.7,
        "max_grad_norm": 0.8,
        "use_sde": False,
        "sde_sample_freq": -1,
        "target_kl": 0.04,
        "stats_window_size": 50,
    }
    assert captured["device"] == "cpu"
    assert captured["total_timesteps"] == 1
    assert [event for event, _fields in events] == [
        "ppo_model_construction_started",
        "ppo_model_constructed",
        "ppo_learn_started",
        "ppo_learn_finished",
    ]


def test_build_continuous_replay_step_returns_contract_replay_shape():
    step = build_continuous_replay_step(
        goal_id="target_goal",
        distance_sensor_id="rangefinder",
        step_duration_seconds=0.1,
        episode_index=0,
        step_index=2,
        reward=0.3,
        info={
            "collision": True,
            "collision_id": "box_001",
            "front_distance": 1.25,
            "camera_mount_height_meters": 0.42,
            "robot_x": 3.0,
            "robot_z": 4.0,
            "robot_rotation_y_degrees": 45.0,
            "applied_forward": 0.6,
            "applied_turn": -0.1,
            "reward_components": [
                {"name": "step_penalty", "value": -0.01},
                {"name": "goal_progress", "value": 0.1},
                {"name": "inactive_penalty", "value": -0.1},
                {"name": "collision_penalty", "value": -50.0},
            ],
        },
        terminated=True,
        truncated=False,
    )

    assert step["phase"] == "eval"
    assert step["checkpoint_step"] == 0
    assert step["env_index"] == 0
    assert step["policy_mode"] == "deterministic"
    assert step["episode_id"] == "eval_env_00_episode_000001"
    assert step["robot"]["position"] == {"x": 3.0, "z": 4.0}
    assert step["robot"]["rotation_y_degrees"] == 45.0
    assert step["action"]["values"] == [
        {"name": "forward", "value": 0.6},
        {"name": "turn", "value": -0.1},
    ]
    assert step["reward"]["components"] == [
        {"name": "step_penalty", "value": -0.01},
        {"name": "goal_progress", "value": 0.1},
        {"name": "inactive_penalty", "value": -0.1},
        {"name": "collision_penalty", "value": -50.0},
    ]
    assert step["terminated"] is True
    assert step["termination_reason"] == "collision"
    assert step["events"] == [
        {
            "type": "collision",
            "object_id": "box_001",
            "message": "Continuous movement was blocked",
        },
    ]
    assert step["sensors"] == [
        {
            "id": "rangefinder",
            "type": "distance_meters",
            "value": 1.25,
        },
        {
            "id": "camera_mount_height",
            "type": "camera_mount_height_meters",
            "value": 0.42,
        },
    ]


def test_training_replay_rejects_inconsistent_sb3_transition_batch():
    reporter = TrainingProgressReporter(
        total_steps=1,
        progress_callback=None,
        replay_writer=object(),
        eval_spec=convert_scenario_to_spec(scenario_bundle()),
    )
    reporter._episode_indices = [0]  # noqa: SLF001
    reporter._episode_steps = [0]  # noqa: SLF001
    reporter.locals = {
        "rewards": np.array([], dtype=np.float32),
        "dones": np.array([], dtype=bool),
        "infos": [{}],
    }

    with pytest.raises(RuntimeError, match="SB3 replay transition batch"):
        reporter._record_training_replay(1)  # noqa: SLF001


def test_training_replay_records_initial_state_for_one_step_episodes():
    class RecordingWriter:
        def __init__(self):
            self.steps = []

        def record_train_step(self, step):
            self.steps.append(step)

    spec = convert_scenario_to_spec(scenario_bundle())
    env = ContinuousNavigationEnv(spec=spec, max_episode_steps=1)
    writer = RecordingWriter()
    _train_model(
        env=env,
        training=training_spec(
            timesteps=8,
            n_steps=8,
            batch_size=4,
            max_episode_steps=1,
        ),
        replay_writer=writer,
        eval_spec=spec,
    )

    assert [step["step_index"] for step in writer.steps] == [0, 1] * 8
    assert [step["time_seconds"] for step in writer.steps] == [0.0, 0.1] * 8
    assert [step["episode_id"] for step in writer.steps] == [
        f"train_env_00_episode_{episode:06d}"
        for episode in range(1, 9)
        for _step in range(2)
    ]
    assert writer.steps[0]["action"]["values"] == [
        {"name": "forward", "value": 0.0},
        {"name": "turn", "value": 0.0},
    ]
    assert writer.steps[0]["reward"] == {"total": 0.0, "components": []}


def test_evaluate_continuous_policy_records_all_eval_episodes(monkeypatch):
    class FakeEnv:
        def __init__(self):
            self.spec = convert_scenario_to_spec(scenario_bundle())
            self.episode_index = -1
            self.step_index = 0

        def reset(self, seed=None):
            self.episode_index += 1
            self.step_index = 0
            obs = {
                "obs_0": np.zeros((1,), dtype=np.float32),
                "obs_1": np.zeros((2,), dtype=np.float32),
            }
            return obs, {
                "front_distance": 5.0,
                "camera_mount_height_meters": 0.6,
                "robot_x": float(self.episode_index),
                "robot_z": 0.0,
                "robot_rotation_y_degrees": 0.0,
                "collision": False,
                "collision_id": None,
                "reward_components": [],
            }

        def step(self, action):
            self.step_index += 1
            terminated = self.step_index >= 2
            info = {
                "front_distance": 5.0,
                "camera_mount_height_meters": 0.6,
                "robot_x": float(self.episode_index),
                "robot_z": float(self.step_index),
                "robot_rotation_y_degrees": 0.0,
                "collision": False,
                "collision_id": None,
                "applied_forward": float(action[0]),
                "applied_turn": float(action[1]),
                "reward_components": [{"name": "step_penalty", "value": -0.01}],
            }
            return (
                {
                    "obs_0": np.zeros((1,), dtype=np.float32),
                    "obs_1": np.zeros((2,), dtype=np.float32),
                },
                1.0,
                terminated,
                False,
                info,
            )

    monkeypatch.setattr(
        "embodiedlab.training.runner._predict_navigation_final_raw_action",
        lambda model, obs: np.array([0.5, 0.0], dtype=np.float32),
    )

    result = evaluate_continuous_policy(
        model=object(),
        env=FakeEnv(),
        training=training_spec(eval_episodes=3),
    )

    assert result["episodes"] == 3
    assert result["success_rate"] == 1.0
    assert [
        (step["episode_id"], step["step_index"]) for step in result["replay_steps"]
    ] == [
        (f"eval_env_00_episode_{episode:06d}", step_index)
        for episode in range(1, 4)
        for step_index in range(3)
    ]
    assert [step["robot"]["position"]["z"] for step in result["replay_steps"]] == [
        0.0,
        1.0,
        2.0,
    ] * 3
    assert {step["phase"] for step in result["replay_steps"]} == {"eval"}
    assert {step["policy_mode"] for step in result["replay_steps"]} == {
        "deterministic",
    }

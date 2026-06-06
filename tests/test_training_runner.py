import numpy as np

from embodiedlab.continuous_navigation_env import ContinuousNavigationEnv
from embodiedlab.schemas import ScenarioBundle
from embodiedlab.training.runner import (
    _build_training_env,
    _predict_navigation_final_raw_action,
    _train_model,
    build_continuous_replay_step,
)
from embodiedlab.training.training_config import TrainingConfig
from embodiedlab.training.training_converter import convert_submission_to_spec


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
        TrainingConfig(torch_num_threads=1),
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


def test_build_training_env_randomizes_start_for_single_env():
    scenario = ScenarioBundle()
    spec = convert_submission_to_spec(scenario)
    training = TrainingConfig(n_envs=1)

    env = _build_training_env(spec=spec, training=training)

    assert env.randomize_start is True


def test_build_training_env_uses_subproc_vec_env_automatically_for_multiple_envs():
    scenario = ScenarioBundle()
    spec = convert_submission_to_spec(scenario)
    training = TrainingConfig(n_envs=2)
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
                "start_method": "fork",
            },
        )
    finally:
        env.close()


def test_navigation_final_raw_prediction_matches_sb3_deterministic_action():
    scenario = ScenarioBundle()
    spec = convert_submission_to_spec(scenario)
    env = ContinuousNavigationEnv(spec=spec, max_steps=10)
    training = TrainingConfig(timesteps=1, n_steps=8, batch_size=4, seed=10)
    model = _train_model(env=env, training=training)
    obs, _info = env.reset(seed=10)

    sb3_action, _state = model.predict(obs, deterministic=True)
    action = _predict_navigation_final_raw_action(model, obs)

    np.testing.assert_allclose(action, sb3_action)


def test_train_model_passes_configured_n_epochs_to_ppo(monkeypatch):
    scenario = ScenarioBundle()
    spec = convert_submission_to_spec(scenario)
    env = ContinuousNavigationEnv(spec=spec, max_steps=10)
    training = TrainingConfig(timesteps=1, n_steps=8, batch_size=4, n_epochs=3)
    captured = {}

    class FakePPO:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def learn(self, total_timesteps, callback=None):
            captured["total_timesteps"] = total_timesteps
            captured["callback"] = callback

    monkeypatch.setattr("embodiedlab.training.runner.PPO", FakePPO)

    model = _train_model(env=env, training=training)

    assert model is not None
    assert captured["n_epochs"] == 3
    assert captured["total_timesteps"] == 1


def test_build_continuous_replay_step_returns_envforge_replay_shape():
    step = build_continuous_replay_step(
        episode_index=0,
        step_index=2,
        action=np.array([0.7, -0.2], dtype=np.float32),
        obs={"obs_1": np.array([30.0, 5.0], dtype=np.float32)},
        reward=0.3,
        info={
            "distance_delta": 1.0,
            "collision": True,
            "collision_id": "box_001",
            "front_distance": 1.25,
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

    assert step["episode_id"] == "episode_0001"
    assert step["robot"]["position"] == {"x": 3.0, "z": 4.0}
    assert step["robot"]["rotation_y_degrees"] == 45.0
    assert step["action"]["values"] == [
        {"name": "forward", "value": 0.6000000238418579},
        {"name": "turn", "value": -0.10000000149011612},
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
            "id": "front_distance",
            "type": "envforge_distance_sensor_meters",
            "value": 1.25,
        },
    ]

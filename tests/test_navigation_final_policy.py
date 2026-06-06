import numpy as np
import torch
from stable_baselines3 import PPO

from embodiedlab.continuous_navigation_env import ContinuousNavigationEnv
from embodiedlab.schemas import ScenarioBundle
from embodiedlab.training.navigation_final_policy import (
    NAVIGATION_FINAL_LOG_STD_INIT,
    NAVIGATION_FINAL_LOG_STD_MAX,
    NAVIGATION_FINAL_LOG_STD_MIN,
    POLICY_ACTION_HIGH,
    POLICY_ACTION_LOW,
    NavigationFinalPolicy,
    navigation_final_contract_action,
)
from embodiedlab.training.training_converter import convert_submission_to_spec


def test_navigation_final_policy_bounds_gaussian_std_before_action_mapping():
    spec = convert_submission_to_spec(ScenarioBundle())
    env = ContinuousNavigationEnv(spec=spec, max_steps=10)
    model = PPO(
        policy=NavigationFinalPolicy,
        env=env,
        n_steps=8,
        batch_size=4,
        seed=10,
    )
    obs, _info = env.reset(seed=10)

    with torch.no_grad():
        model.policy.action_net.weight.zero_()
        model.policy.action_net.bias.copy_(
            torch.tensor([-1_000_000.0, 1_000_000.0]),
        )
        model.policy.log_std.copy_(torch.tensor([10.0, -10.0]))

    distribution = model.policy.get_distribution(
        {
            "obs_0": torch.as_tensor(obs["obs_0"][None], dtype=torch.float32),
            "obs_1": torch.as_tensor(obs["obs_1"][None], dtype=torch.float32),
        },
    )

    assert distribution.distribution.mean.detach().numpy().tolist() == [
        [-1_000_000.0, 1_000_000.0],
    ]
    np.testing.assert_allclose(
        model.policy.log_std.detach().numpy(),
        [NAVIGATION_FINAL_LOG_STD_MAX, NAVIGATION_FINAL_LOG_STD_MIN],
    )


def test_navigation_final_policy_uses_small_initial_raw_action_std():
    spec = convert_submission_to_spec(ScenarioBundle())
    env = ContinuousNavigationEnv(spec=spec, max_steps=10)
    model = PPO(
        policy=NavigationFinalPolicy,
        env=env,
        n_steps=8,
        batch_size=4,
        seed=10,
    )

    np.testing.assert_allclose(
        model.policy.log_std.detach().numpy(),
        [NAVIGATION_FINAL_LOG_STD_INIT, NAVIGATION_FINAL_LOG_STD_INIT],
    )


def test_navigation_final_contract_action_maps_bounded_policy_action():
    raw_actions = torch.tensor(
        [[POLICY_ACTION_LOW - 1.0, -1.5], [0.0, 0.0], [POLICY_ACTION_HIGH + 1.0, 1.5]],
        dtype=torch.float32,
    )

    action = navigation_final_contract_action(raw_actions)

    assert torch.allclose(
        action,
        torch.tensor(
            [
                [0.0, -1.0],
                [0.5, 0.0],
                [1.0, 1.0],
            ],
            dtype=torch.float32,
        ),
    )


def test_navigation_final_expert_actions_stay_inside_raw_policy_contract():
    from embodiedlab.training.navigation_final_expert import (
        _expert_raw_action_from_goal_angle,
    )

    raw_actions = np.asarray(
        [
            _expert_raw_action_from_goal_angle(0.0),
            _expert_raw_action_from_goal_angle(90.0),
            _expert_raw_action_from_goal_angle(-90.0),
        ],
        dtype=np.float32,
    )

    assert np.all(raw_actions[:, 0] >= POLICY_ACTION_LOW)
    assert np.all(raw_actions[:, 0] <= POLICY_ACTION_HIGH)
    assert np.all(raw_actions[:, 1] >= POLICY_ACTION_LOW)
    assert np.all(raw_actions[:, 1] <= POLICY_ACTION_HIGH)
    np.testing.assert_allclose(raw_actions[0], [0.6, 0.0])
    np.testing.assert_allclose(raw_actions[1], [-0.5, 1.0])
    np.testing.assert_allclose(raw_actions[2], [-0.5, -1.0])


def test_navigation_final_expert_reaches_goal_on_wall_corridor_regression_seeds():
    from pathlib import Path

    from embodiedlab.training.navigation_final_expert import _expert_rollout_samples

    scenario = ScenarioBundle.model_validate_json(
        Path(
            "tests/fixtures/envforge/navigation_default_scenario_bundle.json"
        ).read_text(),
    )
    spec = convert_submission_to_spec(scenario)
    regression_seeds = [30, 37, 54, 71, 72, 77, 80, 88, 92, 94]

    for seed in regression_seeds:
        env = ContinuousNavigationEnv(
            spec=spec,
            max_steps=scenario.training.max_episode_steps,
            randomize_start=True,
        )
        samples = _expert_rollout_samples(env=env, seed=seed)
        env.close()

        env = ContinuousNavigationEnv(
            spec=spec,
            max_steps=scenario.training.max_episode_steps,
            randomize_start=True,
        )
        env.reset(seed=seed)
        terminal_info = None
        terminated = False
        truncated = False
        for _obs, action in samples:
            _next_obs, _reward, terminated, truncated, terminal_info = env.step(
                np.asarray(action, dtype=np.float32),
            )
            if terminated or truncated:
                break
        env.close()

        assert terminated is True, seed
        assert truncated is False, seed
        assert terminal_info is not None
        assert terminal_info["collision"] is False, seed

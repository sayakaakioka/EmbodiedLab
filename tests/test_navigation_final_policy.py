import numpy as np
import torch
from stable_baselines3 import PPO

from embodiedlab.continuous_navigation_env import ContinuousNavigationEnv
from embodiedlab.schemas import ScenarioBundle
from embodiedlab.training.navigation_final_policy import (
    NAVIGATION_FINAL_LOG_STD_INIT,
    NAVIGATION_FINAL_LOG_STD_MAX,
    NAVIGATION_FINAL_LOG_STD_MIN,
    POLICY_FORWARD_ACTION_HIGH,
    POLICY_FORWARD_ACTION_LOW,
    POLICY_TURN_ACTION_HIGH,
    POLICY_TURN_ACTION_LOW,
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


def test_navigation_final_policy_uses_ml_agents_strict_initial_raw_action_std():
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


def test_navigation_final_contract_action_matches_ml_agents_strict_mapping():
    raw_actions = torch.tensor(
        [
            [POLICY_FORWARD_ACTION_LOW, POLICY_TURN_ACTION_LOW - 1.0],
            [0.0, 0.0],
            [POLICY_FORWARD_ACTION_HIGH, POLICY_TURN_ACTION_HIGH + 1.0],
        ],
        dtype=torch.float32,
    )

    action = navigation_final_contract_action(raw_actions)

    expected = torch.tensor(
        [
            [torch.sigmoid(torch.tensor(POLICY_FORWARD_ACTION_LOW)).item(), -1.0],
            [0.5, 0.0],
            [torch.sigmoid(torch.tensor(POLICY_FORWARD_ACTION_HIGH)).item(), 1.0],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(action, expected)

"""NavigationFinal policy modules matching the thesis action model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from stable_baselines3.common.distributions import (
    DiagGaussianDistribution,
    Distribution,
    SquashedDiagGaussianDistribution,
)
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

if TYPE_CHECKING:
    from gymnasium import spaces

POLICY_ACTION_LOW = -1.0
POLICY_ACTION_HIGH = 1.0
NAVIGATION_FINAL_LOG_STD_MIN = -5.0
NAVIGATION_FINAL_LOG_STD_MAX = 0.0
NAVIGATION_FINAL_LOG_STD_INIT = -2.0


class SigmoidGateLayer(nn.Module):
    """Sigmoid gate activation used after a fully connected layer."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Gate the fully connected output with its sigmoid activation."""
        return inputs * torch.sigmoid(inputs)


class NavigationFinalFeaturesExtractor(BaseFeaturesExtractor):
    """Extract obs_0 image and obs_1 numeric features as in Figure 2.3."""

    def __init__(self, observation_space: spaces.Dict) -> None:
        """Build the two-branch NavigationFinal feature extractor."""
        super().__init__(observation_space, features_dim=258)
        obs_1_space = observation_space.spaces["obs_1"]
        max_distance = float(obs_1_space.high[1])
        self.register_buffer(
            "max_distance",
            torch.tensor(max(max_distance, 1.0), dtype=torch.float32),
        )
        self.image_branch = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=8, stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(3456, 256),
            nn.LeakyReLU(),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return concatenated image and standardized numeric features."""
        image_features = self.image_branch(observations["obs_0"].float())
        numeric = observations["obs_1"].float()
        angle = numeric[:, 0:1] / 180.0
        distance = (numeric[:, 1:2] / self.max_distance) * 2.0 - 1.0
        numeric_features = torch.cat([angle, distance], dim=1)
        return torch.cat([image_features, numeric_features], dim=1)


class NavigationFinalMlpExtractor(nn.Module):
    """Actor and critic towers with the actor layout from Figure 2.3."""

    latent_dim_pi = 256
    latent_dim_vf = 256

    def __init__(self, feature_dim: int) -> None:
        """Create actor and critic latent networks."""
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(feature_dim, 256),
            SigmoidGateLayer(),
            nn.Linear(256, 256),
            SigmoidGateLayer(),
        )
        self.critic = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return actor and critic latents."""
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        """Return action-model latent features."""
        return self.actor(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        """Return value-model latent features."""
        return self.critic(features)


class NavigationFinalPolicy(MultiInputActorCriticPolicy):
    """Stable-Baselines3 policy matching the NavigationFinal action model."""

    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        """Configure the NavigationFinal feature extractor."""
        kwargs.setdefault(
            "features_extractor_class",
            NavigationFinalFeaturesExtractor,
        )
        kwargs.setdefault("net_arch", [])
        kwargs.setdefault("log_std_init", NAVIGATION_FINAL_LOG_STD_INIT)
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule) -> None:  # noqa: ANN001
        self.action_dist = SquashedDiagGaussianDistribution(
            get_action_dim(self.action_space),
        )
        super()._build(lr_schedule)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = NavigationFinalMlpExtractor(self.features_dim)

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor) -> Distribution:
        """Build a bounded-variance raw-action distribution."""
        if not isinstance(self.action_dist, DiagGaussianDistribution):
            return super()._get_action_dist_from_latent(latent_pi)

        mean_actions = self.action_net(latent_pi)
        self.log_std.data.clamp_(
            NAVIGATION_FINAL_LOG_STD_MIN,
            NAVIGATION_FINAL_LOG_STD_MAX,
        )
        return self.action_dist.proba_distribution(mean_actions, self.log_std)


def navigation_final_contract_action(raw_actions: torch.Tensor) -> torch.Tensor:
    """Map strict raw actions to EnvForge [forward, turn] values."""
    bounded_actions = torch.clamp(raw_actions, POLICY_ACTION_LOW, POLICY_ACTION_HIGH)
    forward = (bounded_actions[..., 0:1] + 1.0) * 0.5
    turn = bounded_actions[..., 1:2]
    return torch.cat([forward, turn], dim=-1)


def navigation_final_deterministic_raw_action(
    policy: MultiInputActorCriticPolicy,
    observations: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Return the deterministic raw Box action from Stable-Baselines3."""
    actions = policy.get_distribution(observations).get_actions(deterministic=True)
    return actions.reshape((-1, *policy.action_space.shape))


def navigation_final_deterministic_action(
    policy: MultiInputActorCriticPolicy,
    observations: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Return EnvForge-ready deterministic [forward, turn] action values."""
    return navigation_final_contract_action(
        navigation_final_deterministic_raw_action(policy, observations),
    )

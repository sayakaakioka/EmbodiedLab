import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from embodiedlab.schemas import (
    MAX_PPO_ROLLOUT_STEPS,
    MAX_RANDOM_SEED,
    MAX_TRAINING_TIMESTEPS,
    DistanceDeltaRewardComponent,
    Position2D,
    ScenarioBundle,
    WorldSpec,
    build_submission_document,
)
from tests.fakes import scenario_bundle, scenario_payload

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _reward_components(**overrides):
    components = scenario_bundle().model_dump(mode="json")["reward"]["components"]
    for component in components:
        component.update(overrides.get(component["name"], {}))
    return components


def test_canonical_scenario_bundle_is_valid():
    scenario = scenario_bundle()
    payload = scenario.model_dump(mode="json")

    assert scenario.schema_version == "scenario-bundle.v0"
    assert scenario.scenario_id == "scenario_demo_001"
    assert scenario.world.coordinate_system == "left_handed_y_up_meters"
    assert scenario.world.goal.position.x == 8.5
    assert scenario.compatibility.robot_version == "simple_robot.v1"
    assert scenario.robot.type == "simple_robot"
    assert scenario.robot.radius == 0.45
    assert scenario.world.goal.radius == scenario.robot.radius
    assert scenario.robot.action_space.layout == ["forward", "turn"]
    assert scenario.robot.action_space.forward_step_meters == 0.2
    assert scenario.robot.action_space.turn_degrees_per_step == 15.0
    assert [sensor.id for sensor in scenario.sensors] == [
        "front_camera",
        "goal_vector",
    ]
    assert "envforge_min_version" not in payload["compatibility"]
    assert payload["world"]["coordinate_system"] == "left_handed_y_up_meters"
    assert scenario.training.algorithm == "ppo"
    assert scenario.training.max_episode_steps == 512


def test_scenario_bundle_requires_all_wire_fields():
    with pytest.raises(ValidationError):
        ScenarioBundle.model_validate({})


def test_scenario_bundle_rejects_deleted_nested_compatibility_field():
    payload = scenario_payload()
    payload["compatibility"]["envforge_min_version"] = "0.1.0"

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ScenarioBundle.model_validate(payload)


def test_camera_resolution_has_a_bounded_contract():
    payload = scenario_payload()
    payload["sensors"][0]["width"] = 513

    with pytest.raises(ValidationError):
        ScenarioBundle.model_validate(payload)


def test_scenario_bundle_rejects_non_finite_numbers():
    payload = scenario_payload()
    payload["robot"]["start_pose"]["rotation_y_degrees"] = float("nan")

    with pytest.raises(ValidationError):
        ScenarioBundle.model_validate(payload)


def test_scenario_bundle_rejects_numeric_string_coercion():
    payload = scenario_payload()
    payload["training"]["timesteps"] = "5000"

    with pytest.raises(ValidationError, match="JSON integer"):
        ScenarioBundle.model_validate(payload)


def test_scenario_bundle_rejects_integer_boolean_coercion():
    payload = scenario_payload()
    payload["training"]["randomize_start"] = 1

    with pytest.raises(ValidationError, match="JSON boolean"):
        ScenarioBundle.model_validate(payload)


def test_training_rejects_timesteps_above_service_limit():
    payload = scenario_payload()
    payload["training"]["timesteps"] = MAX_TRAINING_TIMESTEPS + 1

    with pytest.raises(ValidationError):
        ScenarioBundle.model_validate(payload)


@pytest.mark.parametrize("seed", [-1, MAX_RANDOM_SEED + 1])
def test_training_rejects_seed_outside_uint32_range(seed):
    payload = scenario_payload()
    payload["training"]["seed"] = seed

    with pytest.raises(ValidationError):
        ScenarioBundle.model_validate(payload)


def test_training_rejects_rollout_buffers_above_service_limit():
    payload = scenario_payload()
    payload["training"]["n_envs"] = 32
    payload["training"]["n_steps"] = (MAX_PPO_ROLLOUT_STEPS // 32) + 1
    payload["training"]["batch_size"] = 32

    with pytest.raises(ValidationError, match=r"n_steps \* n_envs"):
        ScenarioBundle.model_validate(payload)


def test_build_submission_document_returns_firestore_payload():
    scenario = scenario_bundle()

    payload = build_submission_document(
        "submission-1",
        scenario,
        cancel_token_hash="a" * 64,
    )

    assert payload["submission_id"] == "submission-1"
    assert isinstance(payload["created_at"], str)
    assert payload["scenario"]["schema_version"] == "scenario-bundle.v0"
    assert payload["scenario"]["scenario_id"] == "scenario_demo_001"
    assert payload["scenario"]["compatibility"]["robot_version"] == "simple_robot.v1"
    assert payload["scenario"]["robot"]["type"] == "simple_robot"
    assert payload["scenario"]["robot"]["radius"] == 0.45
    assert payload["scenario"]["training"]["algorithm"] == "ppo"
    assert payload["control"] == {
        "cancel_token_hash": "a" * 64,
        "dispatch_state": "pending",
        "dispatch_started_at": None,
        "dispatch_error": None,
        "execution_name": None,
        "cancellation_state": "idle",
        "cancellation_started_at": None,
        "cancellation_lease_token": None,
        "cancellation_error": None,
    }


def test_scenario_bundle_accepts_documented_shape():
    payload = scenario_payload()
    payload["scenario_id"] = "scenario_custom"
    payload["world"]["static_obstacles"][0]["id"] = "box_001"
    payload["robot"]["radius"] = 0.3
    payload["world"]["goal"]["radius"] = 0.3
    payload["reward"]["components"] = _reward_components(
        goal_progress={"weight": 0.5},
    )

    scenario = ScenarioBundle.model_validate(payload)

    assert scenario.scenario_id == "scenario_custom"
    assert scenario.world.static_obstacles[0].id == "box_001"
    assert scenario.robot.radius == 0.3
    assert scenario.sensors[0].width == 112
    assert scenario.sensors[0].height == 84
    assert scenario.sensors[0].mount_height_meters == 0.6
    assert scenario.sensors[0].mount_height_min_meters == 0.6
    assert scenario.sensors[0].mount_height_max_meters == 0.6
    assert isinstance(scenario.reward.components[1], DistanceDeltaRewardComponent)


def test_forward_camera_mount_height_range_must_be_complete_and_ordered():
    payload = scenario_payload()
    payload["sensors"][0]["mount_height_min_meters"] = 0.1
    payload["sensors"][0]["mount_height_max_meters"] = None
    with pytest.raises(ValidationError, match="requires both min and max"):
        ScenarioBundle.model_validate(payload)

    payload = scenario_payload()
    payload["sensors"][0]["mount_height_min_meters"] = 1.0
    payload["sensors"][0]["mount_height_max_meters"] = 0.1
    with pytest.raises(ValidationError, match="min must be less than or equal"):
        ScenarioBundle.model_validate(payload)


def test_envforge_navigation_fixture_matches_scenario_bundle_contract():
    fixture_path = FIXTURE_DIR / "envforge" / "navigation_default_scenario_bundle.json"
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))

    scenario = ScenarioBundle.model_validate(payload)

    assert scenario.scenario_id == "navigation_default"
    assert scenario.world.bounds.min.x == -8.0
    assert scenario.world.bounds.max.z == 6.0
    assert scenario.robot.start_pose.position.x == -6.0
    assert scenario.world.goal.radius == 0.45
    assert scenario.world.goal.radius == scenario.robot.radius
    assert scenario.robot.action_space.layout == ["forward", "turn"]
    assert all("height" in wall for wall in payload["world"]["static_walls"])
    assert all(
        "height" in obstacle for obstacle in payload["world"]["static_obstacles"]
    )
    assert [sensor.id for sensor in scenario.sensors] == [
        "front_camera",
        "goal_vector",
    ]
    assert set(payload["sensors"][0]) >= {
        "width",
        "height",
        "semantic_mode",
        "mount_height_meters",
        "mount_height_min_meters",
        "mount_height_max_meters",
        "pitch_degrees",
        "vertical_fov_degrees",
        "near_clip_meters",
        "far_clip_meters",
        "observation_name",
    }
    assert set(payload["sensors"][1]) >= {
        "id",
        "type",
        "target",
        "observation_name",
        "values",
    }
    reward_components = {
        component["name"]: component for component in payload["reward"]["components"]
    }
    assert reward_components["goal_progress"]["minimum_delta_meters"] == 0.005
    assert (
        reward_components["wide_angle_penalty"]["minimum_absolute_angle_degrees"]
        == 90.0
    )
    assert (
        reward_components["rear_angle_penalty"]["minimum_absolute_angle_degrees"]
        == 150.0
    )
    assert reward_components["inactive_penalty"]["maximum_absolute_forward"] == 0.001
    assert scenario.sensors[0].mount_height_min_meters == 0.6
    assert scenario.sensors[0].mount_height_max_meters == 0.6
    assert scenario.training.max_episode_steps == 1000
    assert set(payload["training"]) >= {
        "algorithm",
        "device",
        "timesteps",
        "seed",
        "max_episode_steps",
        "n_envs",
        "cpu_count",
        "torch_num_threads",
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
        "eval_episodes",
        "replay_eval_interval_steps",
        "replay_train_chunk_steps",
        "randomize_start",
    }
    assert scenario.training.n_envs == 1
    assert scenario.training.cpu_count is None
    assert scenario.training.torch_num_threads is None
    assert scenario.training.n_epochs == 3


def test_training_rejects_unknown_field():
    payload = scenario_payload()
    payload["training"]["unsupported_option"] = True
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ScenarioBundle.model_validate(payload)


@pytest.mark.parametrize(
    "world",
    [
        {"bounds": {"min": {"x": 1.0, "z": 0.0}, "max": {"x": 1.0, "z": 2.0}}},
        {"goal": {"id": "goal_001", "position": {"x": 99.0, "z": 1.0}, "radius": 1}},
        {
            "static_obstacles": [
                {
                    "id": "box_001",
                    "shape": "box",
                    "center": {"x": 99.0, "z": 1.0},
                    "size": {"x": 1.0, "z": 1.0},
                }
            ],
        },
    ],
)
def test_world_rejects_invalid_geometry(world):
    payload = scenario_payload()["world"]
    payload.update(world)
    with pytest.raises(ValidationError):
        WorldSpec.model_validate(payload)


def test_scenario_rejects_robot_outside_bounds():
    payload = scenario_payload()
    payload["robot"]["start_pose"]["position"] = {"x": 99.0, "z": 1.0}
    with pytest.raises(ValidationError):
        ScenarioBundle.model_validate(payload)


def test_scenario_rejects_duplicate_sensor_ids():
    payload = scenario_payload()
    payload["sensors"][1]["id"] = payload["sensors"][0]["id"]
    with pytest.raises(ValidationError):
        ScenarioBundle.model_validate(payload)


def test_scenario_rejects_more_than_three_sensors():
    payload = scenario_payload()
    payload["sensors"].extend(
        [
            {
                "id": "front_distance_1",
                "type": "distance_sensor",
                "range_meters": 10.0,
                "direction": "forward",
            },
            {
                "id": "front_distance_2",
                "type": "distance_sensor",
                "range_meters": 10.0,
                "direction": "forward",
            },
        ]
    )

    with pytest.raises(ValidationError):
        ScenarioBundle.model_validate(payload)


def test_world_rejects_more_than_128_combined_geometry_elements():
    payload = scenario_payload()["world"]
    wall = {
        "id": "wall",
        "center": {"x": 1.0, "z": 1.0},
        "size": {"x": 0.5, "z": 0.5},
        "height": 2.0,
        "rotation_y_degrees": 0.0,
    }
    obstacle = payload["static_obstacles"][0]
    payload["static_walls"] = [
        {**wall, "id": f"wall_{index:03d}"} for index in range(64)
    ]
    payload["static_obstacles"] = [
        {**obstacle, "id": f"obstacle_{index:03d}"} for index in range(65)
    ]

    with pytest.raises(ValidationError, match="together cannot exceed 128"):
        WorldSpec.model_validate(payload)


def test_scenario_rejects_missing_reward_target():
    payload = scenario_payload()
    payload["reward"]["components"] = _reward_components(
        goal_progress={"target": "missing_goal"},
    )
    with pytest.raises(ValidationError):
        ScenarioBundle.model_validate(payload)


def test_scenario_rejects_missing_reward_component():
    payload = scenario_payload()
    payload["reward"]["components"] = _reward_components()[:-1]
    with pytest.raises(ValidationError, match="at least 7"):
        ScenarioBundle.model_validate(payload)


def test_scenario_rejects_duplicate_reward_component():
    components = _reward_components()
    components[-1] = components[0]
    payload = scenario_payload()
    payload["reward"]["components"] = components

    with pytest.raises(ValidationError, match="must be unique"):
        ScenarioBundle.model_validate(payload)


def test_scenario_rejects_unknown_reward_component():
    payload = scenario_payload()
    payload["reward"]["components"] = _reward_components(
        inactive_penalty={"name": "custom_reward"},
    )
    with pytest.raises(ValidationError, match="unknown="):
        ScenarioBundle.model_validate(payload)


def test_scenario_rejects_wrong_reward_component_type():
    payload = scenario_payload()
    payload["reward"]["components"] = _reward_components(
        goal_reached={"type": "per_step"},
    )
    with pytest.raises(ValidationError, match="must use type terminal_reward"):
        ScenarioBundle.model_validate(payload)


def test_reward_angle_thresholds_must_be_strictly_ordered():
    payload = scenario_payload()
    for component in payload["reward"]["components"]:
        if component["name"] == "wide_angle_penalty":
            component["minimum_absolute_angle_degrees"] = 150.0

    with pytest.raises(ValidationError, match="threshold must be less"):
        ScenarioBundle.model_validate(payload)


def test_training_rejects_eval_replay_over_sdk_row_limit():
    payload = scenario_payload()
    payload["training"]["max_episode_steps"] = 1001
    payload["training"]["eval_episodes"] = 100
    with pytest.raises(
        ValidationError,
        match=r"eval_episodes \* max_episode_steps",
    ):
        ScenarioBundle.model_validate(payload)


def test_action_layout_is_fixed_for_initial_robot():
    payload = scenario_payload()
    payload["robot"]["action_space"]["layout"] = ["turn", "forward"]
    with pytest.raises(ValidationError):
        ScenarioBundle.model_validate(payload)


def test_position_model_uses_xz_plane():
    position = Position2D(x=1.0, z=2.0)

    assert position.model_dump() == {"x": 1.0, "z": 2.0}

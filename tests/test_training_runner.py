import numpy as np

from embodiedlab.training.runner import build_continuous_replay_step


def test_build_continuous_replay_step_returns_envforge_replay_shape():
    step = build_continuous_replay_step(
        episode_index=0,
        step_index=2,
        action=np.array([0.7, -0.2], dtype=np.float32),
        obs={"robot": np.array([3.0, 4.0, 45.0], dtype=np.float32)},
        reward=0.3,
        info={
            "distance_delta": 1.0,
            "collision": True,
            "collision_id": "box_001",
            "front_distance": 1.25,
        },
        terminated=False,
        truncated=False,
    )

    assert step["episode_id"] == "episode_0001"
    assert step["robot"]["position"] == {"x": 3.0, "z": 4.0}
    assert step["robot"]["rotation_y_degrees"] == 45.0
    assert step["action"]["values"] == [
        {"name": "forward", "value": 0.699999988079071},
        {"name": "turn", "value": -0.20000000298023224},
    ]
    assert step["reward"]["components"] == [
        {"name": "step_penalty", "value": -0.01},
        {"name": "goal_progress", "value": 0.5},
        {"name": "collision_penalty", "value": -5.0},
    ]
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

import numpy as np

from embodiedlab.training.runner import build_grid_replay_step


def test_build_grid_replay_step_returns_envforge_replay_shape():
    step = build_grid_replay_step(
        episode_index=0,
        step_index=2,
        action=1,
        obs={"agent": np.array([3, 4])},
        reward=0.3,
        info={"distance": 5, "distance_delta": 1, "blocked": True},
        terminated=False,
        truncated=False,
    )

    assert step["episode_id"] == "episode_0001"
    assert step["robot"]["position"] == {"x": 3.0, "z": 4.0}
    assert step["robot"]["rotation_y_degrees"] == 90.0
    assert step["action"]["values"] == [
        {"name": "forward", "value": 0.0},
        {"name": "turn", "value": 1.0},
    ]
    assert step["reward"]["components"] == [
        {"name": "step_penalty", "value": -0.2},
        {"name": "goal_progress", "value": 0.5},
    ]
    assert step["events"] == [
        {
            "type": "collision",
            "object_id": None,
            "message": "Grid movement was blocked",
        },
    ]
    assert step["sensors"] == [
        {
            "id": "front_distance",
            "type": "grid_manhattan_distance",
            "value": 5.0,
        },
    ]

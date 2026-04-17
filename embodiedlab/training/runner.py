from __future__ import annotations

from pathlib import Path
from statistics import mean

from stable_baselines3 import PPO

from embodiedlab.gridworld_env import GridWorldTrainingEnv
from embodiedlab.training.training_config import TrainingAlgorithm, TrainingConfig


def evaluate_policy(model: PPO, env: GridWorldTrainingEnv, training: TrainingConfig) -> dict:
	rewards: list[float] = []
	steps: list[int] = []
	successes = 0

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

		if terminated:
			successes += 1

		rewards.append(episode_reward)
		steps.append(episode_steps)

	return {
		"episodes": training.eval_episodes,
		"success_rate": successes / training.eval_episodes,
		"avg_reward": mean(rewards),
		"avg_steps": mean(steps),
	}


def run_gridworld_training(
	spec,
	training: TrainingConfig,
	model_output_path: str | None = None,
) -> dict:
	if training.algorithm is not TrainingAlgorithm.PPO:
		raise ValueError(f"Unsupported algorithm: {training.algorithm}")

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

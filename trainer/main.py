from __future__ import annotations

from trainer.config import load_trainer_config
from trainer.job import run_training_job

def main() -> None:
	config = load_trainer_config()
	run_training_job(config)

if __name__ == "__main__":
	main()

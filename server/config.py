from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ServerConfig:
	db_id: str
	region: str
	job_path: str | None


def load_server_config() -> ServerConfig:
	return ServerConfig(
		db_id=os.environ.get("DB_ID", "(default)"),
		region=os.environ.get("REGION", "asia-northeast1"),
		job_path=os.environ.get("JOB_PATH"),
	)

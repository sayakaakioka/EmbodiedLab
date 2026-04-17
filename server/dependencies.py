from __future__ import annotations

from typing import Any

from google.cloud import firestore

from server.config import ServerConfig, load_server_config

config = load_server_config()
_db = None


def get_config() -> ServerConfig:
	return config


def get_db() -> Any:
	global _db
	if _db is None:
		_db = firestore.Client(database=config.db_id)

	return _db

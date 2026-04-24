"""Pub/Sub payload decoding helpers for the notification service."""

from __future__ import annotations

import base64
import json
from typing import Any

from fastapi import HTTPException
from pydantic import ValidationError

from embodiedlab.result_models import parse_result_message


def decode_pubsub_event(body: dict[str, Any]) -> dict[str, Any]:
    """Decode the Pub/Sub push payload into an event dictionary."""
    if "message" not in body:
        raise HTTPException(status_code=400, detail="Invalid Pub/Sub message")

    pubsub_message = body["message"]
    data_b64 = pubsub_message.get("data")
    if not data_b64:
        raise HTTPException(status_code=400, detail="Missing data")

    try:
        decoded = base64.b64decode(data_b64).decode("utf-8")
        return parse_result_message(json.loads(decoded))
    except ValidationError as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid result event",
        ) from exc
    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid encoded event",
        ) from exc

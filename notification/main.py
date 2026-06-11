"""WebSocket fan-out service: receives Pub/Sub push events and relays to subscribers."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from google.cloud import firestore

from notification.pubsub import decode_pubsub_event
from notification.registry import ConnectionRegistry

ResultFetcher = Callable[[str], dict[str, Any] | None]

logger = logging.getLogger(__name__)


def create_firestore_result_fetcher(db_id: str | None) -> ResultFetcher:
    """Create a Firestore-backed result snapshot fetcher."""
    if not db_id:
        return lambda _submission_id: None

    def fetch_result(submission_id: str) -> dict[str, Any] | None:
        db = firestore.Client(database=db_id)
        result_snap = db.collection("results").document(submission_id).get()
        if not result_snap.exists:
            return None
        return result_snap.to_dict()

    return fetch_result


def create_app(result_fetcher: ResultFetcher | None = None) -> FastAPI:
    """Create the notification service application."""
    app = FastAPI()
    app.state.connections = ConnectionRegistry()
    app.state.result_fetcher = result_fetcher or create_firestore_result_fetcher(
        os.getenv("DB_ID"),
    )

    @app.websocket("/ws/results/{submission_id}")
    async def ws_results(websocket: WebSocket, submission_id: str) -> None:
        """Accept a WebSocket connection and relay training events to the client."""
        connections: ConnectionRegistry = app.state.connections
        await websocket.accept()
        connections.add(submission_id, websocket)
        logger.info(
            "result websocket connected submission_id=%s connections=%d",
            submission_id,
            connections.count(submission_id),
        )

        try:
            await websocket.send_json(
                {
                    "type": "connected",
                    "submission_id": submission_id,
                },
            )

            try:
                latest_result = app.state.result_fetcher(submission_id)
            except Exception:
                logger.exception(
                    "failed to fetch latest result snapshot submission_id=%s",
                    submission_id,
                )
            else:
                if latest_result is not None:
                    await websocket.send_json(latest_result)
                    logger.info(
                        "sent latest result snapshot submission_id=%s status=%s",
                        submission_id,
                        latest_result.get("status"),
                    )
                else:
                    logger.info(
                        "no latest result snapshot submission_id=%s",
                        submission_id,
                    )

            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            connections.remove(submission_id, websocket)
            logger.info(
                "result websocket disconnected submission_id=%s connections=%d",
                submission_id,
                connections.count(submission_id),
            )

    @app.post("/internal/pubsub/push")
    async def pubsub_push(request: Request) -> dict[str, str]:
        """Receive a Pub/Sub push message and fan it out to subscribed clients."""
        event = decode_pubsub_event(await request.json())
        submission_id = event.get("submission_id")
        if not submission_id:
            raise HTTPException(status_code=400, detail="Missing submission_id")

        connections: ConnectionRegistry = app.state.connections
        event_status = event.get("status", "unknown")
        logger.info(
            "pubsub result event received submission_id=%s status=%s",
            submission_id,
            event_status,
        )
        stats = await connections.broadcast(submission_id, event)
        logger.info(
            "pubsub result event broadcast submission_id=%s status=%s "
            "sent=%d dead=%d remaining=%d",
            submission_id,
            event_status,
            stats["sent"],
            stats["dead"],
            stats["remaining"],
        )
        return {"status": "ok"}

    return app


app = create_app()

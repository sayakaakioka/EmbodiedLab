"""WebSocket fan-out service: receives Pub/Sub push events and relays to subscribers."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect

from notification.pubsub import decode_pubsub_event
from notification.registry import ConnectionRegistry


def create_app() -> FastAPI:
    """Create the notification service application."""
    app = FastAPI()
    app.state.connections = ConnectionRegistry()

    @app.websocket("/ws/results/{submission_id}")
    async def ws_results(websocket: WebSocket, submission_id: str) -> None:
        """Accept a WebSocket connection and relay training events to the client."""
        connections: ConnectionRegistry = app.state.connections
        await websocket.accept()
        connections.add(submission_id, websocket)

        try:
            await websocket.send_json(
                {
                    "type": "connected",
                    "submission_id": submission_id,
                },
            )

            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            connections.remove(submission_id, websocket)

    @app.post("/internal/pubsub/push")
    async def pubsub_push(request: Request) -> dict[str, str]:
        """Receive a Pub/Sub push message and fan it out to subscribed clients."""
        event = decode_pubsub_event(await request.json())
        submission_id = event.get("submission_id")
        if not submission_id:
            raise HTTPException(status_code=400, detail="Missing submission_id")

        connections: ConnectionRegistry = app.state.connections
        await connections.broadcast(submission_id, event)
        return {"status": "ok"}

    return app


app = create_app()

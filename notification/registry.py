"""In-memory WebSocket connection registry for notification fan-out."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import WebSocket


class ConnectionRegistry:
    """Track WebSocket subscriptions by submission ID."""

    def __init__(self) -> None:
        """Initialize the empty connection registry."""
        self._connections: defaultdict[str, set[WebSocket]] = defaultdict(set)

    def add(self, submission_id: str, websocket: WebSocket) -> None:
        """Register a WebSocket for a submission."""
        self._connections[submission_id].add(websocket)

    def remove(self, submission_id: str, websocket: WebSocket) -> None:
        """Remove a WebSocket and clean up empty submission buckets."""
        if websocket in self._connections[submission_id]:
            self._connections[submission_id].remove(websocket)
        if not self._connections[submission_id]:
            del self._connections[submission_id]

    async def broadcast(self, submission_id: str, event: dict) -> None:
        """Send an event to all registered clients for the submission."""
        dead_connections: list[WebSocket] = []
        for websocket in list(self._connections.get(submission_id, [])):
            try:
                await websocket.send_json(event)
            except Exception:  # noqa: BLE001
                dead_connections.append(websocket)

        for websocket in dead_connections:
            self._connections[submission_id].discard(websocket)

        if submission_id in self._connections and not self._connections[submission_id]:
            del self._connections[submission_id]

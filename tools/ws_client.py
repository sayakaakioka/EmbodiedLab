import asyncio
import os

import websockets
from websockets.exceptions import ConnectionClosed


def get_required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        message = f"{name} is required"
        raise RuntimeError(message)
    return value


def build_url() -> str:
    service_name = get_required_env("NOTIFICATION_SERVICE_NAME")
    hash_value = get_required_env("HASH")
    region = get_required_env("REGION")
    submission_id = get_required_env("SUBMISSION_ID")

    return (
        f"wss://{service_name}-{hash_value}.{region}.run.app"
        f"/ws/results/{submission_id}"
    )


async def main():
    url = build_url()

    async with websockets.connect(url) as ws:
        print("connected")
        try:
            async for msg in ws:
                print(msg)
        except ConnectionClosed as exc:
            print(f"closed: {exc}")
        except asyncio.CancelledError:
            pass


try:
    asyncio.run(main())
except KeyboardInterrupt:
    print("\nclosed")

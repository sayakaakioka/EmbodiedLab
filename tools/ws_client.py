import asyncio
import os

import websockets

submission_id = os.environ["SUBMISSION_ID"]

url = f"wss://embodiedlab-notification-886092613885.asia-northeast1.run.app/ws/results/{submission_id}"


async def main():
    async with websockets.connect(url) as ws:
        print("connected")
        try:
            async for msg in ws:
                print(msg)
        except asyncio.CancelledError:
            pass


try:
    asyncio.run(main())
except KeyboardInterrupt:
    print("\nclosed")

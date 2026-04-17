from __future__ import annotations

from fastapi import FastAPI

from server.routes import router


def create_app() -> FastAPI:
	app = FastAPI()
	app.include_router(router)
	return app


app = create_app()

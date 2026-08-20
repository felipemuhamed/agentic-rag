from fastapi import FastAPI

from app.api.chat import router as chat_router
from app.api.health import router as health_router
from app.settings import settings


def create_app() -> FastAPI:
    application = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
    )
    application.include_router(health_router)
    application.include_router(chat_router)
    return application


app = create_app()

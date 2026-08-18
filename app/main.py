from fastapi import FastAPI

from app.api.health import router as health_router
from app.settings import settings


def create_app() -> FastAPI:
    application = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
    )
    application.include_router(health_router)
    return application


app = create_app()

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi import APIRouter
from pydantic import BaseModel
from contextlib import asynccontextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.modules.services.database import Base
from app.tasks import analyze_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 API iniciada")
    yield
    print("🛑 API detenida")

def run_app() -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    app.include_router(analyze_router)
    return app
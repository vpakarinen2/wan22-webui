from __future__ import annotations

import os

from pydantic import BaseModel
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(override=True)

_PROJECT_ROOT_DOTENV = Path(__file__).resolve().parents[2] / ".env"
if _PROJECT_ROOT_DOTENV.exists():
    load_dotenv(dotenv_path=_PROJECT_ROOT_DOTENV, override=True)


class Settings(BaseModel):
    port: int = int(os.getenv("WAN_UI_PORT", "7860"))
    queue_concurrency: int = int(os.getenv("WAN_UI_QUEUE_CONCURRENCY", "1"))
    models_root: str = os.getenv("WAN_MODELS_ROOT", "/workspace/models")
    cache_root: str = os.getenv("WAN_CACHE_ROOT", "/workspace/cache")
    outputs_root: str = os.getenv("WAN_OUTPUTS_ROOT", "outputs")
    logs_root: str = os.getenv("WAN_LOGS_ROOT", "logs")
    execute_enabled: bool = os.getenv("WAN_EXECUTE", "0") in {"1", "true", "True"}
    tts_enabled: bool = os.getenv("WAN_TTS_ENABLED", "0") in {"1", "true", "True"}


settings = Settings()

import os
from pathlib import Path

from dotenv import load_dotenv


def load_api_key(dotenv_path: Path | str | None = None) -> str:
    load_dotenv(dotenv_path)
    API_KEY = os.getenv("API_KEY")
    if API_KEY is None:
        raise ValueError("Missing API_KEY in .env file")
    return API_KEY

"""Local OpenAI/Ollama compatibility gateway for the general LFM service."""

from .app import create_app
from .config import Settings

__all__ = ["Settings", "create_app"]
__version__ = "1.0.0"

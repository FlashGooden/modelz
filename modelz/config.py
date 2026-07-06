import os
from pathlib import Path

from .errors import ConfigError


def load_api_token(env_path: Path | None = None) -> str:
    """Load REPLICATE_API_TOKEN from a .env file (if present) and the environment."""
    env_path = env_path or Path(".env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

    token = os.environ.get("REPLICATE_API_TOKEN")
    if not token:
        raise ConfigError(
            "REPLICATE_API_TOKEN is not set. Copy .env.example to .env and add your token."
        )
    return token

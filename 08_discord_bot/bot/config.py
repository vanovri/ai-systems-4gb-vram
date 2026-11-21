"""Environment-based configuration and fail-fast Discord token validation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_PLACEHOLDER_TOKENS = frozenset(
    {
        "",
        "your_discord_bot_token_here",
        "your_token_here",
        "changeme",
        "placeholder",
        "<discord_token>",
        "replace_me",
    }
)


class ConfigurationError(RuntimeError):
    """Raised when required runtime configuration is missing or invalid."""


def load_dotenv_file(env_path: str | Path = ".env") -> dict[str, str]:
    """Minimal zero-dependency .env loader (does not overwrite existing env vars)."""
    path = Path(env_path)
    loaded: dict[str, str] = {}
    if not path.exists() or not path.is_file():
        return loaded

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded[key] = value
    return loaded


def validate_token(token: str | None) -> str:
    """Fail-fast validation ensuring DISCORD_TOKEN is set and non-placeholder."""
    cleaned = (token or "").strip()
    if not cleaned or cleaned.lower() in _PLACEHOLDER_TOKENS:
        raise ConfigurationError(
            "DISCORD_TOKEN is missing or still set to a placeholder. "
            "Set a valid token via the DISCORD_TOKEN environment variable or .env file."
        )
    if len(cleaned) < 20 or any(ch.isspace() for ch in cleaned):
        raise ConfigurationError(
            "DISCORD_TOKEN appears malformed (must be >= 20 non-whitespace characters)."
        )
    return cleaned


@dataclass(frozen=True)
class BotConfig:
    """Immutable runtime configuration loaded strictly from environment variables."""

    token: str
    command_prefix: str = "!"
    reminder_db_path: Path = Path("data/reminders.db")
    log_level: str = "INFO"
    cooldown_seconds: float = 5.0
    http_timeout_seconds: float = 10.0
    github_token: str | None = None


def load_config(
    *,
    require_token: bool = True,
    env_file: str | Path | None = None,
) -> BotConfig:
    """Load BotConfig from environment (and optional .env file)."""
    if env_file is not None:
        load_dotenv_file(env_file)

    raw_token = os.getenv("DISCORD_TOKEN", "").strip()
    token = validate_token(raw_token) if require_token else raw_token

    prefix = os.getenv("COMMAND_PREFIX", "!").strip() or "!"
    db_path = Path(os.getenv("REMINDER_DB_PATH", "data/reminders.db"))
    log_level = os.getenv("LOG_LEVEL", "INFO").upper().strip() or "INFO"

    try:
        cooldown_seconds = float(os.getenv("COOLDOWN_SECONDS", "5.0"))
        http_timeout_seconds = float(os.getenv("HTTP_TIMEOUT_SECONDS", "10.0"))
    except ValueError as exc:
        raise ConfigurationError(f"Numeric environment variable invalid: {exc}") from exc

    if cooldown_seconds < 0 or http_timeout_seconds <= 0:
        raise ConfigurationError("COOLDOWN_SECONDS must be >= 0 and HTTP_TIMEOUT_SECONDS > 0.")

    github_token = os.getenv("GITHUB_TOKEN", "").strip() or None

    return BotConfig(
        token=token,
        command_prefix=prefix,
        reminder_db_path=db_path,
        log_level=log_level,
        cooldown_seconds=cooldown_seconds,
        http_timeout_seconds=http_timeout_seconds,
        github_token=github_token,
    )

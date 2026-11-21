"""Per-user command cooldown / rate-limit guard."""

from __future__ import annotations

import time


class CooldownExceededError(Exception):
    """Raised when a user invokes a rate-limited command during cooldown."""

    def __init__(self, command_name: str, retry_after: float) -> None:
        self.command_name = command_name
        self.retry_after = max(0.0, float(retry_after))
        super().__init__(
            f"Command '{command_name}' is on cooldown. Try again in {self.retry_after:.1f}s."
        )


class PerUserCooldown:
    """In-memory per-(command, user) rate limiter with injectable monotonic clock."""

    def __init__(self, default_cooldown: float = 5.0) -> None:
        if default_cooldown < 0:
            raise ValueError("default_cooldown must be non-negative")
        self.default_cooldown = float(default_cooldown)
        self._last_used: dict[tuple[str, int], float] = {}

    def check_and_consume(
        self,
        command_name: str,
        user_id: int,
        cooldown_seconds: float | None = None,
        now: float | None = None,
    ) -> float:
        """Consume a command slot or raise CooldownExceededError."""
        cd = self.default_cooldown if cooldown_seconds is None else float(cooldown_seconds)
        if cd <= 0:
            return 0.0

        current = time.monotonic() if now is None else float(now)
        key = (command_name, int(user_id))
        last = self._last_used.get(key)
        if last is not None:
            elapsed = current - last
            if elapsed < cd:
                raise CooldownExceededError(command_name=command_name, retry_after=cd - elapsed)

        self._last_used[key] = current
        return 0.0

    def reset(self, command_name: str | None = None, user_id: int | None = None) -> None:
        if command_name is None and user_id is None:
            self._last_used.clear()
            return
        to_delete = [
            key
            for key in self._last_used
            if (command_name is None or key[0] == command_name)
            and (user_id is None or key[1] == user_id)
        ]
        for key in to_delete:
            del self._last_used[key]

"""Offline async mock-gateway test suite for Project #8 — Modular Discord Bot."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import discord
import httpx
import pytest
from discord.ext import commands

from bot.cogs.automation import AutomationCog
from bot.cogs.general import GeneralCog, get_process_memory_mb
from bot.cogs.moderation import ModerationCog
from bot.config import BotConfig, ConfigurationError, load_config, validate_token
from bot.logging_utils import JsonLogFormatter
from bot.main import ModularDiscordBot, main as bot_cli_main, run_offline_demo
from bot.rate_limit import CooldownExceededError, PerUserCooldown
from bot.services import ExternalServiceError, ResourceNotFoundError
from bot.storage import ReminderStore


class MockChannel:
    """In-memory Discord text channel capturing sent messages and Embeds."""

    def __init__(self, channel_id: int = 9001) -> None:
        self.id = channel_id
        self.sent: list[dict[str, Any]] = []

    async def send(
        self, content: str | None = None, *, embed: discord.Embed | None = None
    ) -> SimpleNamespace:
        record = {"content": content, "embed": embed}
        self.sent.append(record)
        return SimpleNamespace(id=len(self.sent), content=content, embed=embed)


class MockContext:
    """Offline mock discord.ext.commands.Context."""

    def __init__(
        self,
        *,
        author_id: int = 12345,
        author_name: str = "tester",
        channel: MockChannel | None = None,
        command_name: str = "test",
    ) -> None:
        self.author = SimpleNamespace(
            id=author_id,
            name=author_name,
            mention=f"<@{author_id}>",
            bot=False,
        )
        self.channel = channel or MockChannel()
        self.guild = SimpleNamespace(id=55555, name="MockGuild")
        self.command = SimpleNamespace(qualified_name=command_name)
        self.sent = self.channel.sent

    async def send(
        self, content: str | None = None, *, embed: discord.Embed | None = None
    ) -> SimpleNamespace:
        return await self.channel.send(content=content, embed=embed)


def _build_mock_transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "geocoding-api.open-meteo.com" in url:
            city = request.url.params.get("name", "")
            if city.lower() == "atlantis":
                return httpx.Response(200, json={"results": []})
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "name": "Jakarta",
                            "country": "Indonesia",
                            "latitude": -6.2,
                            "longitude": 106.8,
                        }
                    ]
                },
            )
        if "api.open-meteo.com" in url:
            return httpx.Response(
                200,
                json={
                    "current": {
                        "temperature_2m": 30.5,
                        "relative_humidity_2m": 74,
                        "wind_speed_10m": 12.3,
                        "weather_code": 2,
                    }
                },
            )
        if "api.github.com/repos/python/cpython" in url:
            return httpx.Response(
                200,
                json={
                    "full_name": "python/cpython",
                    "description": "The Python programming language",
                    "stargazers_count": 64123,
                    "forks_count": 30456,
                    "open_issues_count": 8100,
                    "language": "Python",
                    "license": {"spdx_id": "PSF-2.0"},
                    "html_url": "https://github.com/python/cpython",
                    "updated_at": "2026-11-20T00:00:00Z",
                },
            )
        if "api.github.com/repos/missing/repo" in url:
            return httpx.Response(404, json={"message": "Not Found"})
        if "api.github.com/repos/ratelimited/repo" in url:
            return httpx.Response(403, json={"message": "API rate limit exceeded"})
        return httpx.Response(500, json={"error": "unexpected"})

    return httpx.MockTransport(handler)


@pytest.fixture
async def mock_bot(tmp_path: Path):
    cfg = BotConfig(
        token="",
        command_prefix="!",
        reminder_db_path=tmp_path / "reminders.db",
        cooldown_seconds=5.0,
    )
    client = httpx.AsyncClient(transport=_build_mock_transport())
    bot = ModularDiscordBot(cfg, http_client=client)
    await bot.load_default_cogs()
    try:
        yield bot
    finally:
        await bot.close()
        await client.aclose()


def test_config_fail_fast_rejects_missing_and_placeholder_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DISCORD_TOKEN", raising=False)
    with pytest.raises(ConfigurationError, match="missing or still set to a placeholder"):
        load_config(require_token=True)

    with pytest.raises(ConfigurationError):
        validate_token("your_discord_bot_token_here")
    with pytest.raises(ConfigurationError):
        validate_token("short")

    valid = validate_token("A" * 32)
    assert valid == "A" * 32


def test_json_log_formatter_emits_structured_json() -> None:
    formatter = JsonLogFormatter()
    record = logging.LogRecord(
        name="discord_bot.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="Command executed",
        args=(),
        exc_info=None,
    )
    record.event = "command_ok"
    record.user_id = 42
    parsed = json.loads(formatter.format(record))
    assert parsed["level"] == "INFO"
    assert parsed["event"] == "command_ok"
    assert parsed["user_id"] == 42
    assert parsed["message"] == "Command executed"


def test_sqlite_reminder_store_persistence_and_lifecycle(tmp_path: Path) -> None:
    db_file = tmp_path / "persist.db"
    store1 = ReminderStore(db_file)
    rec = store1.add_reminder(
        user_id=77, channel_id=88, message="Check deployment", delay_seconds=30.0, now=1000.0
    )
    assert rec.id >= 1
    assert store1.count_pending() == 1

    # Re-open fresh instance simulating bot restart
    store2 = ReminderStore(db_file)
    due_early = store2.get_due(now=1020.0)
    assert due_early == []
    due_now = store2.get_due(now=1031.0)
    assert len(due_now) == 1
    assert due_now[0].message == "Check deployment"

    assert store2.mark_fired(rec.id, now=1031.0) is True
    assert store2.count_pending() == 0
    assert store2.get_by_id(rec.id).fired is True

    # Verify cancel works for unfired reminders
    rec2 = store2.add_reminder(
        user_id=77, channel_id=88, message="Cancelable", delay_seconds=60.0, now=1100.0
    )
    assert store2.cancel_reminder(rec2.id, user_id=999) is False
    assert store2.cancel_reminder(rec2.id, user_id=77) is True
    assert store2.count_pending() == 0


def test_per_user_cooldown_guard_enforces_and_expires() -> None:
    cd = PerUserCooldown(default_cooldown=5.0)
    assert cd.check_and_consume("weather", 101, now=100.0) == 0.0
    # Different user allowed immediately
    assert cd.check_and_consume("weather", 202, now=100.5) == 0.0

    # Same user blocked during cooldown window
    with pytest.raises(CooldownExceededError) as exc_info:
        cd.check_and_consume("weather", 101, now=102.0)
    assert pytest.approx(exc_info.value.retry_after, abs=1e-6) == 3.0

    # Same user allowed after cooldown expires
    assert cd.check_and_consume("weather", 101, now=105.1) == 0.0


async def test_general_cog_hello_ping_and_status(mock_bot: ModularDiscordBot) -> None:
    cog: GeneralCog = mock_bot.get_cog("general")  # type: ignore[assignment]
    ctx = MockContext(author_id=1001)

    await cog.hello.callback(cog, ctx)
    assert "<@1001>" in ctx.sent[0]["content"]

    await cog.ping.callback(cog, ctx)
    ping_embed: discord.Embed = ctx.sent[1]["embed"]
    assert ping_embed.title == "Pong!"
    assert "ms" in (ping_embed.description or "")

    await cog.status.callback(cog, ctx)
    status_embed: discord.Embed = ctx.sent[2]["embed"]
    assert status_embed.title == "Bot Operational Status"
    field_names = {f.name for f in status_embed.fields}
    assert {"Uptime", "Gateway Latency", "Memory RSS", "Loaded Cogs", "Pending Reminders"} <= field_names


async def test_automation_weather_embed_formatting(mock_bot: ModularDiscordBot) -> None:
    cog: AutomationCog = mock_bot.get_cog("automation")  # type: ignore[assignment]
    ctx = MockContext(author_id=2001, command_name="weather")

    await cog.weather.callback(cog, ctx, city="Jakarta")
    assert len(ctx.sent) == 1
    embed: discord.Embed = ctx.sent[0]["embed"]
    assert embed.title == "Weather — Jakarta, Indonesia"
    assert "Partly cloudy" in (embed.description or "")
    fields = {f.name: f.value for f in embed.fields}
    assert fields["Temperature"] == "30.5 °C"
    assert fields["Humidity"] == "74%"
    assert fields["Wind"] == "12.3 km/h"


async def test_automation_weather_not_found_raises(mock_bot: ModularDiscordBot) -> None:
    cog: AutomationCog = mock_bot.get_cog("automation")  # type: ignore[assignment]
    ctx = MockContext(author_id=2002, command_name="weather")

    with pytest.raises(ResourceNotFoundError, match="Atlantis"):
        await cog.weather.callback(cog, ctx, city="Atlantis")


async def test_automation_github_embed_and_error_cases(mock_bot: ModularDiscordBot) -> None:
    cog: AutomationCog = mock_bot.get_cog("automation")  # type: ignore[assignment]
    ctx = MockContext(author_id=3001, command_name="github")

    await cog.github.callback(cog, ctx, repo="python/cpython")
    embed: discord.Embed = ctx.sent[0]["embed"]
    assert embed.title == "GitHub — python/cpython"
    assert embed.url == "https://github.com/python/cpython"
    fields = {f.name: f.value for f in embed.fields}
    assert fields["Stars"] == "64,123"
    assert fields["Language"] == "Python"
    assert fields["License"] == "PSF-2.0"

    # Invalid slug validation
    ctx2 = MockContext(author_id=3002, command_name="github")
    with pytest.raises(ValueError, match="owner/repo"):
        await cog.github.callback(cog, ctx2, repo="not-a-valid-slug")

    # 404 Not Found
    ctx3 = MockContext(author_id=3003, command_name="github")
    with pytest.raises(ResourceNotFoundError):
        await cog.github.callback(cog, ctx3, repo="missing/repo")

    # 403 Rate Limit
    ctx4 = MockContext(author_id=3004, command_name="github")
    with pytest.raises(ExternalServiceError, match="rate limit"):
        await cog.github.callback(cog, ctx4, repo="ratelimited/repo")


async def test_automation_remind_persists_and_fires_async(mock_bot: ModularDiscordBot) -> None:
    cog: AutomationCog = mock_bot.get_cog("automation")  # type: ignore[assignment]
    ctx = MockContext(author_id=4001, command_name="remind")

    await cog.remind.callback(cog, ctx, seconds=0.05, message="Run unit tests")
    assert len(ctx.sent) == 1
    assert "Scheduled" in (ctx.sent[0]["embed"].title or "")
    assert mock_bot.reminder_store.count_pending() == 1

    await asyncio.sleep(0.15)
    assert len(ctx.sent) == 2
    assert "Reminder due!" in (ctx.sent[1]["content"] or "")
    assert ctx.sent[1]["embed"].description == "Run unit tests"
    assert mock_bot.reminder_store.count_pending() == 0


async def test_automation_command_cooldown_triggers_error(mock_bot: ModularDiscordBot) -> None:
    cog: AutomationCog = mock_bot.get_cog("automation")  # type: ignore[assignment]
    ctx = MockContext(author_id=5001, command_name="weather")

    await cog.weather.callback(cog, ctx, city="Jakarta")
    with pytest.raises(CooldownExceededError) as exc_info:
        await cog.weather.callback(cog, ctx, city="Jakarta")
    assert exc_info.value.command_name == "weather"
    assert exc_info.value.retry_after > 0


async def test_moderation_audit_events_ready_and_message(mock_bot: ModularDiscordBot) -> None:
    mod: ModerationCog = mock_bot.get_cog("moderation")  # type: ignore[assignment]
    await mod.on_ready()

    user_msg = SimpleNamespace(
        author=SimpleNamespace(id=7001, bot=False),
        guild=SimpleNamespace(id=8001),
        channel=SimpleNamespace(id=9001),
        content="!hello",
    )
    bot_msg = SimpleNamespace(
        author=SimpleNamespace(id=9999, bot=True),
        guild=SimpleNamespace(id=8001),
        channel=SimpleNamespace(id=9001),
        content="ignored bot message",
    )
    await mod.on_message(user_msg)
    await mod.on_message(bot_msg)

    event_types = [e["event"] for e in mod.audit_events]
    assert "bot_ready" in event_types
    assert event_types.count("message_received") == 1
    msg_event = [e for e in mod.audit_events if e["event"] == "message_received"][0]
    assert msg_event["is_command"] is True
    assert msg_event["author_id"] == 7001


async def test_moderation_on_command_error_produces_clean_embeds(
    mock_bot: ModularDiscordBot,
) -> None:
    mod: ModerationCog = mock_bot.get_cog("moderation")  # type: ignore[assignment]
    ctx = MockContext(author_id=6001, command_name="weather")

    await mod.on_command_error(ctx, commands.CommandNotFound("Command 'foo' is not found"))
    assert ctx.sent[-1]["embed"].title == "Unknown Command"

    await mod.on_command_error(
        ctx, commands.CommandInvokeError(CooldownExceededError("weather", 3.4))
    )
    assert ctx.sent[-1]["embed"].title == "Rate Limit / Cooldown Active"
    assert "3.4s" in (ctx.sent[-1]["embed"].description or "")

    await mod.on_command_error(
        ctx, commands.CommandInvokeError(ResourceNotFoundError("City 'X' not found"))
    )
    assert ctx.sent[-1]["embed"].title == "Resource Not Found"

    await mod.on_command_error(ctx, commands.CommandInvokeError(RuntimeError("boom")))
    assert ctx.sent[-1]["embed"].title == "Internal Command Error"
    assert "boom" not in (ctx.sent[-1]["embed"].description or "")


async def test_offline_demo_runs_cleanly() -> None:
    rc = await run_offline_demo()
    assert rc == 0


def test_cli_check_fails_fast_without_token_and_memory_positive(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("DISCORD_TOKEN", raising=False)
    rc_fail = bot_cli_main(["--check", "--env-file", str(tmp_path / "missing.env")])
    assert rc_fail == 2

    monkeypatch.setenv("DISCORD_TOKEN", "X" * 32)
    rc_ok = bot_cli_main(["--check", "--env-file", str(tmp_path / "missing.env")])
    assert rc_ok == 0
    assert get_process_memory_mb() > 0.0


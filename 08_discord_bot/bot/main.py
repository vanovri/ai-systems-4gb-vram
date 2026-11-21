"""Discord Bot main entrypoint with modular Cog loading and graceful shutdown."""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import discord
import httpx
from discord.ext import commands

from bot.cogs.automation import AutomationCog
from bot.cogs.general import GeneralCog
from bot.cogs.moderation import ModerationCog
from bot.config import BotConfig, ConfigurationError, load_config
from bot.logging_utils import configure_logging
from bot.rate_limit import PerUserCooldown
from bot.storage import ReminderStore

logger = logging.getLogger("discord_bot.main")


class ModularDiscordBot(commands.Bot):
    """Production-ready commands.Bot managing shared HTTP client, SQLite store, and Cogs."""

    def __init__(
        self,
        config: BotConfig,
        *,
        http_client: httpx.AsyncClient | None = None,
        reminder_store: ReminderStore | None = None,
        cooldown: PerUserCooldown | None = None,
    ) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix=config.command_prefix, intents=intents)

        self.config = config
        self.started_at = time.monotonic()
        self._owns_http_client = http_client is None
        self.http_client = http_client or httpx.AsyncClient(timeout=config.http_timeout_seconds)
        self.reminder_store = reminder_store or ReminderStore(config.reminder_db_path)
        self.cooldown = cooldown or PerUserCooldown(default_cooldown=config.cooldown_seconds)

    async def load_default_cogs(self) -> None:
        """Load general, automation, and moderation cogs."""
        if "general" not in self.cogs:
            await self.add_cog(GeneralCog(self))
        if "automation" not in self.cogs:
            await self.add_cog(AutomationCog(self))
        if "moderation" not in self.cogs:
            await self.add_cog(ModerationCog(self))

    async def setup_hook(self) -> None:
        await self.load_default_cogs()
        logger.info(
            "All modular cogs loaded",
            extra={"event": "setup_hook_complete", "cogs": sorted(self.cogs.keys())},
        )

    async def close(self) -> None:
        """Gracefully unload cogs, cancel pending reminder tasks, and close HTTP pools."""
        logger.info("Shutting down bot gracefully", extra={"event": "bot_shutdown_start"})
        for cog_name in list(self.cogs.keys()):
            await self.remove_cog(cog_name)
        if self._owns_http_client and not self.http_client.is_closed:
            await self.http_client.aclose()
        await super().close()
        logger.info("Bot shutdown complete", extra={"event": "bot_shutdown_complete"})


async def run_bot(config: BotConfig) -> None:
    """Start the bot and register graceful SIGINT/SIGTERM shutdown handlers."""
    bot = ModularDiscordBot(config)
    loop = asyncio.get_running_loop()

    def _request_shutdown(sig_name: str) -> None:
        logger.info(
            "Received shutdown signal",
            extra={"event": "signal_received", "signal": sig_name},
        )
        if not bot.is_closed():
            asyncio.create_task(bot.close())

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_shutdown, sig.name)
        except (NotImplementedError, RuntimeError):
            # Windows ProactorEventLoop falls back to KeyboardInterrupt handling
            pass

    try:
        async with bot:
            await bot.start(config.token)
    except KeyboardInterrupt:
        if not bot.is_closed():
            await bot.close()


async def run_offline_demo() -> int:
    """Run an end-to-end offline mock-gateway demonstration without a live Discord token."""
    import tempfile
    from types import SimpleNamespace

    class _DemoChannel:
        def __init__(self, cid: int) -> None:
            self.id = cid
            self.messages: list[dict[str, Any]] = []

        async def send(self, content: str | None = None, *, embed: discord.Embed | None = None) -> None:
            self.messages.append({"content": content, "embed": embed})

    class _DemoContext:
        def __init__(self, author_id: int, channel: _DemoChannel) -> None:
            self.author = SimpleNamespace(id=author_id, mention=f"<@{author_id}>", bot=False)
            self.channel = channel
            self.command = SimpleNamespace(qualified_name="demo")

        async def send(self, content: str | None = None, *, embed: discord.Embed | None = None) -> None:
            await self.channel.send(content=content, embed=embed)

    def _mock_handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "geocoding-api.open-meteo.com" in url:
            return httpx.Response(
                200,
                json={"results": [{"name": "Tokyo", "country": "Japan", "latitude": 35.68, "longitude": 139.76}]},
            )
        if "api.open-meteo.com" in url:
            return httpx.Response(
                200,
                json={
                    "current": {
                        "temperature_2m": 22.4,
                        "relative_humidity_2m": 58,
                        "wind_speed_10m": 11.2,
                        "weather_code": 1,
                    }
                },
            )
        if "api.github.com/repos/" in url:
            return httpx.Response(
                200,
                json={
                    "full_name": "python/cpython",
                    "description": "The Python programming language",
                    "stargazers_count": 65000,
                    "forks_count": 31000,
                    "open_issues_count": 8200,
                    "language": "Python",
                    "license": {"spdx_id": "PSF-2.0"},
                    "html_url": "https://github.com/python/cpython",
                    "updated_at": "2026-11-20T00:00:00Z",
                },
            )
        return httpx.Response(404)

    with tempfile.TemporaryDirectory() as tmp:
        cfg = BotConfig(
            token="",
            command_prefix="!",
            reminder_db_path=Path(tmp) / "demo_reminders.db",
            cooldown_seconds=5.0,
        )
        async with httpx.AsyncClient(transport=httpx.MockTransport(_mock_handler)) as client:
            bot = ModularDiscordBot(cfg, http_client=client)
            await bot.load_default_cogs()
            ch = _DemoChannel(1001)
            ctx = _DemoContext(author_id=42, channel=ch)

            general: GeneralCog = bot.get_cog("general")  # type: ignore[assignment]
            automation: AutomationCog = bot.get_cog("automation")  # type: ignore[assignment]
            moderation: ModerationCog = bot.get_cog("moderation")  # type: ignore[assignment]

            await general.hello.callback(general, ctx)
            await general.ping.callback(general, ctx)
            await general.status.callback(general, ctx)
            await automation.weather.callback(automation, ctx, city="Tokyo")
            await automation.github.callback(automation, ctx, repo="python/cpython")
            await automation.remind.callback(automation, ctx, seconds=0.05, message="Drink water!")
            await asyncio.sleep(0.12)

            # Demonstrate cooldown error handling through moderation cog
            try:
                await automation.weather.callback(automation, ctx, city="Tokyo")
            except Exception as exc:
                ctx.command = SimpleNamespace(qualified_name="weather")
                await moderation.on_command_error(ctx, exc)

            await bot.close()

            print("\n=== Offline Mock-Gateway Demo Summary ===")
            for idx, msg in enumerate(ch.messages, start=1):
                embed = msg["embed"]
                if embed is not None:
                    fields = ", ".join(f"{f.name}={f.value}" for f in embed.fields)
                    print(f"[{idx}] EMBED: {embed.title} | {embed.description or ''} | {fields}")
                else:
                    print(f"[{idx}] TEXT:  {msg['content']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="discord-bot",
        description="Modular 24/7 Discord Bot (general, automation, moderation cogs).",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Optional .env path loaded before reading environment variables.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate configuration and token fail-fast without connecting to Discord.",
    )
    parser.add_argument(
        "--demo-offline",
        action="store_true",
        help="Run offline mock-gateway demo exercising all commands, embeds, SQLite, and cooldowns.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging()

    if args.demo_offline:
        return asyncio.run(run_offline_demo())

    try:
        config = load_config(require_token=True, env_file=args.env_file)
    except ConfigurationError as exc:
        logger.error(str(exc), extra={"event": "config_error"})
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    configure_logging(config.log_level)
    if args.check:
        logger.info("Configuration valid", extra={"event": "config_check_ok"})
        print("Configuration OK (token validated, ready to connect).")
        return 0

    asyncio.run(run_bot(config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

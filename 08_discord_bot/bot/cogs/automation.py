"""Automation Cog: /weather, /github, /remind with SQLite persistence and per-user cooldown."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

from discord.ext import commands

from bot.services import (
    GitHubClient,
    OpenMeteoClient,
    build_github_embed,
    build_reminder_embed,
    build_weather_embed,
)
from bot.storage import ReminderRecord

if TYPE_CHECKING:
    from bot.main import ModularDiscordBot

logger = logging.getLogger("discord_bot.automation")

MAX_REMINDER_SECONDS = 7 * 24 * 3600  # 7 days


class AutomationCog(commands.Cog, name="automation"):
    """Async API automations and persistent SQLite reminder scheduler."""

    def __init__(
        self,
        bot: ModularDiscordBot,
        weather_client: OpenMeteoClient | None = None,
        github_client: GitHubClient | None = None,
    ) -> None:
        self.bot = bot
        self.weather_client = weather_client or OpenMeteoClient(bot.http_client)
        self.github_client = github_client or GitHubClient(
            bot.http_client, token=bot.config.github_token
        )
        self._active_tasks: dict[int, asyncio.Task[None]] = {}

    async def cog_unload(self) -> None:
        """Cancel all in-flight reminder tasks cleanly on shutdown."""
        tasks = list(self._active_tasks.values())
        self._active_tasks.clear()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _enforce_cooldown(self, command_name: str, user_id: int) -> None:
        self.bot.cooldown.check_and_consume(command_name, user_id)

    async def _deliver_reminder(
        self,
        record: ReminderRecord,
        delay: float,
        channel: Any,
        mention: str,
    ) -> None:
        try:
            if delay > 0:
                await asyncio.sleep(delay)
            embed = build_reminder_embed(record, fired=True)
            if channel is not None:
                await channel.send(content=f"{mention} Reminder due!", embed=embed)
            self.bot.reminder_store.mark_fired(record.id)
            logger.info(
                "Reminder fired",
                extra={
                    "event": "reminder_fired",
                    "reminder_id": record.id,
                    "user_id": record.user_id,
                    "channel_id": record.channel_id,
                },
            )
        except asyncio.CancelledError:
            raise
        finally:
            self._active_tasks.pop(record.id, None)

    async def process_due_reminders(self, now: float | None = None) -> int:
        """Recover and fire any due SQLite reminders (for startup recovery or periodic sweep)."""
        due_rows = self.bot.reminder_store.get_due(now=now)
        fired_count = 0
        for rec in due_rows:
            channel = self.bot.get_channel(rec.channel_id)
            if channel is not None:
                embed = build_reminder_embed(rec, fired=True)
                await channel.send(content=f"<@{rec.user_id}> Reminder due!", embed=embed)
            if self.bot.reminder_store.mark_fired(rec.id, now=now):
                fired_count += 1
        return fired_count

    @commands.hybrid_command(
        name="weather", description="Fetch live current weather for a city via Open-Meteo."
    )
    async def weather(self, ctx: commands.Context, *, city: str) -> None:
        self._enforce_cooldown("weather", ctx.author.id)
        report = await self.weather_client.fetch_weather(city)
        embed = build_weather_embed(report)
        await ctx.send(embed=embed)
        logger.info(
            "Weather command served",
            extra={
                "event": "command_weather",
                "user_id": ctx.author.id,
                "city": report.city,
                "temp_c": report.temperature_c,
            },
        )

    @commands.hybrid_command(
        name="github", description="Show public GitHub repository metrics (owner/repo)."
    )
    async def github(self, ctx: commands.Context, repo: str) -> None:
        self._enforce_cooldown("github", ctx.author.id)
        info = await self.github_client.fetch_repo(repo)
        embed = build_github_embed(info)
        await ctx.send(embed=embed)
        logger.info(
            "GitHub command served",
            extra={
                "event": "command_github",
                "user_id": ctx.author.id,
                "repo": info.full_name,
                "stars": info.stars,
            },
        )

    @commands.hybrid_command(
        name="remind", description="Schedule a persistent reminder: /remind <seconds> <message>"
    )
    async def remind(self, ctx: commands.Context, seconds: float, *, message: str) -> None:
        self._enforce_cooldown("remind", ctx.author.id)
        if seconds <= 0 or seconds > MAX_REMINDER_SECONDS:
            raise ValueError(
                f"Reminder delay must be between 0.1 and {MAX_REMINDER_SECONDS} seconds."
            )
        cleaned_msg = message.strip()
        if not cleaned_msg:
            raise ValueError("Reminder message cannot be empty.")

        channel_id = int(getattr(ctx.channel, "id", 0))
        record = self.bot.reminder_store.add_reminder(
            user_id=int(ctx.author.id),
            channel_id=channel_id,
            message=cleaned_msg,
            delay_seconds=seconds,
            now=time.time(),
        )
        embed = build_reminder_embed(record, fired=False)
        await ctx.send(embed=embed)

        mention = getattr(ctx.author, "mention", f"<@{ctx.author.id}>")
        task = asyncio.create_task(
            self._deliver_reminder(
                record=record,
                delay=seconds,
                channel=ctx.channel,
                mention=mention,
            ),
            name=f"reminder-{record.id}",
        )
        self._active_tasks[record.id] = task
        logger.info(
            "Reminder scheduled",
            extra={
                "event": "reminder_scheduled",
                "reminder_id": record.id,
                "user_id": ctx.author.id,
                "delay_seconds": seconds,
            },
        )


async def setup(bot: ModularDiscordBot) -> None:
    await bot.add_cog(AutomationCog(bot))

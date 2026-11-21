"""Moderation & Audit Cog: on_ready, on_message, and clean on_command_error handling."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from discord.ext import commands

from bot.rate_limit import CooldownExceededError
from bot.services import ExternalServiceError, ResourceNotFoundError, build_error_embed

if TYPE_CHECKING:
    from bot.main import ModularDiscordBot

logger = logging.getLogger("discord_bot.moderation")


class ModerationCog(commands.Cog, name="moderation"):
    """Structured audit event listeners and clean user-facing error handling."""

    def __init__(self, bot: ModularDiscordBot) -> None:
        self.bot = bot
        self.audit_events: list[dict[str, Any]] = []

    def _record_audit(self, event: str, **fields: Any) -> dict[str, Any]:
        entry = {"event": event, **fields}
        self.audit_events.append(entry)
        logger.info(f"Audit event: {event}", extra=entry)
        return entry

    @commands.Cog.listener()
    async def on_ready(self) -> None:
        user_str = str(self.bot.user) if self.bot.user is not None else "unknown"
        guild_count = len(getattr(self.bot, "guilds", []))
        self._record_audit(
            "bot_ready",
            bot_user=user_str,
            guild_count=guild_count,
            cogs=sorted(self.bot.cogs.keys()),
        )

    @commands.Cog.listener()
    async def on_message(self, message: Any) -> None:
        author = getattr(message, "author", None)
        if author is None or getattr(author, "bot", False):
            return

        guild = getattr(message, "guild", None)
        channel = getattr(message, "channel", None)
        content = str(getattr(message, "content", ""))
        is_command = content.startswith(self.bot.config.command_prefix)

        self._record_audit(
            "message_received",
            author_id=getattr(author, "id", None),
            guild_id=getattr(guild, "id", None) if guild is not None else None,
            channel_id=getattr(channel, "id", None) if channel is not None else None,
            content_length=len(content),
            is_command=is_command,
        )

    @commands.Cog.listener()
    async def on_command_error(self, ctx: commands.Context, error: Exception) -> None:
        """Convert internal or command errors into clean Embed messages and audit logs."""
        original = getattr(error, "original", error)
        cmd_name = getattr(getattr(ctx, "command", None), "qualified_name", "unknown")

        if isinstance(error, commands.CommandNotFound):
            title = "Unknown Command"
            detail = (
                f"Command not recognized. Use `{self.bot.config.command_prefix}hello`, "
                "`/ping`, `/status`, `/weather`, `/github`, or `/remind`."
            )
            error_type = "command_not_found"
        elif isinstance(error, commands.MissingRequiredArgument):
            title = "Missing Required Argument"
            detail = f"Missing required parameter `{error.param.name}` for command `{cmd_name}`."
            error_type = "missing_argument"
        elif isinstance(error, commands.BadArgument):
            title = "Invalid Argument"
            detail = f"Invalid argument provided for `{cmd_name}`: {error}"
            error_type = "bad_argument"
        elif isinstance(original, CooldownExceededError):
            title = "Rate Limit / Cooldown Active"
            detail = (
                f"Slow down! `{original.command_name}` is on cooldown. "
                f"Please retry in **{original.retry_after:.1f}s**."
            )
            error_type = "cooldown_exceeded"
        elif isinstance(error, commands.CommandOnCooldown):
            title = "Rate Limit / Cooldown Active"
            detail = f"Please wait **{error.retry_after:.1f}s** before using `{cmd_name}` again."
            error_type = "cooldown_exceeded"
        elif isinstance(original, ResourceNotFoundError):
            title = "Resource Not Found"
            detail = str(original)
            error_type = "not_found"
        elif isinstance(original, ExternalServiceError):
            title = "Upstream Service Unavailable"
            detail = str(original)
            error_type = "external_service_error"
        elif isinstance(original, ValueError):
            title = "Validation Error"
            detail = str(original)
            error_type = "validation_error"
        else:
            title = "Internal Command Error"
            detail = "An unexpected error occurred while executing this command. It has been logged."
            error_type = type(original).__name__

        self._record_audit(
            "command_error",
            command=cmd_name,
            user_id=getattr(getattr(ctx, "author", None), "id", None),
            error_type=error_type,
            detail=detail,
        )
        embed = build_error_embed(title=title, detail=detail)
        await ctx.send(embed=embed)


async def setup(bot: ModularDiscordBot) -> None:
    await bot.add_cog(ModerationCog(bot))

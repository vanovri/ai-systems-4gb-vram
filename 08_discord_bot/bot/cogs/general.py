"""General Cog: !hello, /ping, and /status commands."""

from __future__ import annotations

import ctypes
import platform
import sys
import time
from typing import TYPE_CHECKING

import discord
from discord.ext import commands

from bot.services import build_status_embed

if TYPE_CHECKING:
    from bot.main import ModularDiscordBot


def get_process_memory_mb() -> float:
    """Return current process RSS memory in MB cross-platform without third-party dependencies."""
    if sys.platform == "win32":
        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        get_current_process = ctypes.windll.kernel32.GetCurrentProcess
        get_current_process.restype = ctypes.c_void_p
        get_mem_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_mem_info.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
        get_mem_info.restype = ctypes.c_int

        handle = get_current_process()
        if get_mem_info(handle, ctypes.byref(counters), counters.cb):
            return counters.WorkingSetSize / (1024 * 1024)
        return 0.0

    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS returns bytes; Linux returns kilobytes
        divisor = (1024 * 1024) if sys.platform == "darwin" else 1024
        return float(usage) / divisor
    except Exception:
        return 0.0


class GeneralCog(commands.Cog, name="general"):
    """Core liveness and telemetry commands."""

    def __init__(self, bot: ModularDiscordBot) -> None:
        self.bot = bot

    @commands.command(name="hello", help="Greet the invoking user.")
    async def hello(self, ctx: commands.Context) -> None:
        mention = getattr(ctx.author, "mention", str(ctx.author))
        await ctx.send(f"Hello {mention}! Bot is online and ready.")

    @commands.hybrid_command(name="ping", description="Check gateway WebSocket latency.")
    async def ping(self, ctx: commands.Context) -> None:
        raw_latency = getattr(self.bot, "latency", 0.0)
        latency_ms = 0.0 if raw_latency != raw_latency else round(raw_latency * 1000.0, 1)
        embed = discord.Embed(
            title="Pong!",
            description=f"Gateway latency: **{latency_ms:.1f} ms**",
            color=discord.Color.green(),
        )
        await ctx.send(embed=embed)

    @commands.hybrid_command(
        name="status", description="Show bot uptime, latency, memory usage, and active cogs."
    )
    async def status(self, ctx: commands.Context) -> None:
        started_at = getattr(self.bot, "started_at", time.monotonic())
        uptime_seconds = max(0.0, time.monotonic() - started_at)
        raw_latency = getattr(self.bot, "latency", 0.0)
        latency_ms = 0.0 if raw_latency != raw_latency else round(raw_latency * 1000.0, 1)
        memory_mb = get_process_memory_mb()
        loaded_cogs = sorted(self.bot.cogs.keys())
        pending_reminders = (
            self.bot.reminder_store.count_pending()
            if hasattr(self.bot, "reminder_store") and self.bot.reminder_store is not None
            else 0
        )

        embed = build_status_embed(
            uptime_seconds=uptime_seconds,
            latency_ms=latency_ms,
            memory_mb=memory_mb,
            loaded_cogs=loaded_cogs,
            pending_reminders=pending_reminders,
            python_version=platform.python_version(),
            discord_version=discord.__version__,
        )
        await ctx.send(embed=embed)


async def setup(bot: ModularDiscordBot) -> None:
    await bot.add_cog(GeneralCog(bot))

"""Async HTTP API clients (Open-Meteo Weather, GitHub Repos) and Discord Embed formatters."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import discord
import httpx

from bot.storage import ReminderRecord

REPO_SLUG_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")

WMO_WEATHER_CODES: dict[int, str] = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    95: "Thunderstorm",
}


class ExternalServiceError(RuntimeError):
    """Raised when an upstream API request fails or returns unexpected data."""


class ResourceNotFoundError(ExternalServiceError):
    """Raised when a requested city or repository does not exist."""


@dataclass(frozen=True)
class WeatherReport:
    city: str
    country: str
    latitude: float
    longitude: float
    temperature_c: float
    humidity_pct: int
    wind_speed_kmh: float
    weather_code: int
    condition: str


@dataclass(frozen=True)
class GitHubRepoInfo:
    full_name: str
    description: str
    stars: int
    forks: int
    open_issues: int
    language: str
    license_name: str
    html_url: str
    updated_at: str


class OpenMeteoClient:
    """Async Open-Meteo weather client (public API, requires zero API keys)."""

    GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
    FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

    def __init__(self, http_client: httpx.AsyncClient) -> None:
        self._client = http_client

    async def fetch_weather(self, city: str) -> WeatherReport:
        query = city.strip()
        if not query:
            raise ValueError("City name cannot be empty.")

        try:
            geo_resp = await self._client.get(
                self.GEOCODING_URL,
                params={"name": query, "count": 1, "language": "en", "format": "json"},
            )
            geo_resp.raise_for_status()
            geo_data = geo_resp.json()
        except httpx.HTTPError as exc:
            raise ExternalServiceError(f"Weather geocoding failed: {exc}") from exc

        results = geo_data.get("results") or []
        if not results:
            raise ResourceNotFoundError(f"City '{query}' not found in Open-Meteo geocoding.")

        place = results[0]
        lat = float(place["latitude"])
        lon = float(place["longitude"])
        resolved_city = str(place.get("name", query))
        country = str(place.get("country", "Unknown"))

        try:
            wx_resp = await self._client.get(
                self.FORECAST_URL,
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
                },
            )
            wx_resp.raise_for_status()
            wx_data = wx_resp.json()
        except httpx.HTTPError as exc:
            raise ExternalServiceError(f"Weather forecast failed: {exc}") from exc

        current = wx_data.get("current") or {}
        if "temperature_2m" not in current:
            raise ExternalServiceError("Open-Meteo response missing 'current.temperature_2m'.")

        code = int(current.get("weather_code", 0))
        return WeatherReport(
            city=resolved_city,
            country=country,
            latitude=lat,
            longitude=lon,
            temperature_c=float(current["temperature_2m"]),
            humidity_pct=int(current.get("relative_humidity_2m", 0)),
            wind_speed_kmh=float(current.get("wind_speed_10m", 0.0)),
            weather_code=code,
            condition=WMO_WEATHER_CODES.get(code, f"Weather code {code}"),
        )


class GitHubClient:
    """Async GitHub public repository metadata client."""

    BASE_URL = "https://api.github.com/repos"

    def __init__(self, http_client: httpx.AsyncClient, token: str | None = None) -> None:
        self._client = http_client
        self._token = token

    async def fetch_repo(self, slug: str) -> GitHubRepoInfo:
        cleaned = slug.strip().strip("/")
        if not REPO_SLUG_RE.match(cleaned):
            raise ValueError("Repository must be in 'owner/repo' format (e.g. python/cpython).")

        headers: dict[str, str] = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "aartii-discord-bot/1.0",
        }
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"

        try:
            resp = await self._client.get(f"{self.BASE_URL}/{cleaned}", headers=headers)
        except httpx.HTTPError as exc:
            raise ExternalServiceError(f"GitHub API request failed: {exc}") from exc

        if resp.status_code == 404:
            raise ResourceNotFoundError(f"GitHub repository '{cleaned}' not found.")
        if resp.status_code == 403:
            raise ExternalServiceError("GitHub API rate limit exceeded. Configure GITHUB_TOKEN.")
        if resp.status_code >= 400:
            raise ExternalServiceError(f"GitHub API returned HTTP {resp.status_code}.")

        data: dict[str, Any] = resp.json()
        license_info = data.get("license") or {}
        return GitHubRepoInfo(
            full_name=str(data.get("full_name", cleaned)),
            description=str(data.get("description") or "No description provided."),
            stars=int(data.get("stargazers_count", 0)),
            forks=int(data.get("forks_count", 0)),
            open_issues=int(data.get("open_issues_count", 0)),
            language=str(data.get("language") or "Unknown"),
            license_name=str(license_info.get("spdx_id") or license_info.get("name") or "None"),
            html_url=str(data.get("html_url", f"https://github.com/{cleaned}")),
            updated_at=str(data.get("updated_at", "unknown")),
        )


def build_weather_embed(report: WeatherReport) -> discord.Embed:
    """Format WeatherReport as a rich Discord Embed."""
    embed = discord.Embed(
        title=f"Weather — {report.city}, {report.country}",
        description=f"**{report.condition}**",
        color=discord.Color.blue(),
        timestamp=datetime.now(timezone.utc),
    )
    embed.add_field(name="Temperature", value=f"{report.temperature_c:.1f} °C", inline=True)
    embed.add_field(name="Humidity", value=f"{report.humidity_pct}%", inline=True)
    embed.add_field(name="Wind", value=f"{report.wind_speed_kmh:.1f} km/h", inline=True)
    embed.set_footer(text=f"Open-Meteo ({report.latitude:.2f}, {report.longitude:.2f})")
    return embed


def build_github_embed(info: GitHubRepoInfo) -> discord.Embed:
    """Format GitHubRepoInfo as a rich Discord Embed."""
    embed = discord.Embed(
        title=f"GitHub — {info.full_name}",
        url=info.html_url,
        description=info.description[:300],
        color=discord.Color.dark_teal(),
        timestamp=datetime.now(timezone.utc),
    )
    embed.add_field(name="Stars", value=f"{info.stars:,}", inline=True)
    embed.add_field(name="Forks", value=f"{info.forks:,}", inline=True)
    embed.add_field(name="Open Issues", value=f"{info.open_issues:,}", inline=True)
    embed.add_field(name="Language", value=info.language, inline=True)
    embed.add_field(name="License", value=info.license_name, inline=True)
    embed.set_footer(text=f"Updated: {info.updated_at}")
    return embed


def build_status_embed(
    *,
    uptime_seconds: float,
    latency_ms: float,
    memory_mb: float,
    loaded_cogs: list[str],
    pending_reminders: int,
    python_version: str,
    discord_version: str,
) -> discord.Embed:
    """Format bot health metrics as a Discord Embed."""
    hours, rem = divmod(int(uptime_seconds), 3600)
    minutes, secs = divmod(rem, 60)
    uptime_str = f"{hours:02d}h {minutes:02d}m {secs:02d}s"

    embed = discord.Embed(
        title="Bot Operational Status",
        color=discord.Color.green(),
        timestamp=datetime.now(timezone.utc),
    )
    embed.add_field(name="Uptime", value=uptime_str, inline=True)
    embed.add_field(name="Gateway Latency", value=f"{latency_ms:.1f} ms", inline=True)
    embed.add_field(name="Memory RSS", value=f"{memory_mb:.1f} MB", inline=True)
    embed.add_field(name="Pending Reminders", value=str(pending_reminders), inline=True)
    embed.add_field(name="Loaded Cogs", value=", ".join(loaded_cogs) or "none", inline=False)
    embed.set_footer(text=f"Python {python_version} | discord.py {discord_version}")
    return embed


def build_reminder_embed(record: ReminderRecord, *, fired: bool = False) -> discord.Embed:
    """Format scheduled or triggered reminder as a Discord Embed."""
    if fired:
        embed = discord.Embed(
            title=f"Reminder #{record.id}",
            description=record.message,
            color=discord.Color.gold(),
            timestamp=datetime.now(timezone.utc),
        )
        embed.set_footer(text="Scheduled reminder delivered")
        return embed

    delay = max(0, round(record.due_at - record.created_at, 1))
    embed = discord.Embed(
        title=f"Reminder #{record.id} Scheduled",
        description=f"I will remind you in **{delay}s**:\n> {record.message}",
        color=discord.Color.blurple(),
        timestamp=datetime.now(timezone.utc),
    )
    embed.set_footer(text="Persisted in SQLite")
    return embed


def build_error_embed(title: str, detail: str) -> discord.Embed:
    """Format user-friendly error message as a red Discord Embed."""
    embed = discord.Embed(
        title=title,
        description=detail,
        color=discord.Color.red(),
        timestamp=datetime.now(timezone.utc),
    )
    return embed

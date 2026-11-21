# Project #8 - Modular Discord Bot (`08_discord_bot`)

Production-grade `discord.py` v2 modular bot with Cog architecture, async `httpx` external API automations (Open-Meteo Weather & GitHub Repo metrics), SQLite-persisted reminder scheduling, per-user rate-limit cooldowns, structured JSON logging, graceful shutdown, and a 24/7 Docker deployment pack.

## Features

- **Modular Cog Architecture (`bot/cogs/`)**:
  - `general.py`: `!hello`, `/ping` (WebSocket gateway latency Embed), `/status` (uptime, latency, process RSS memory MB, loaded cogs, pending SQLite reminders).
  - `automation.py`:
    - `/weather <city>`: Async Open-Meteo Geocoding + Forecast client (public API, **zero API key required**) formatted as a Discord Embed.
    - `/github <owner/repo>`: Async GitHub public API repository summary Embed (stars, forks, open issues, primary language, license, URL; optional `GITHUB_TOKEN` for higher rate limits).
    - `/remind <seconds> <message>`: Async reminder scheduler backed by lightweight **SQLite WAL persistence** (`bot/storage.py`) with startup/periodic due-reminder recovery.
    - **Per-user cooldown guard** (`bot/rate_limit.py`) protecting external API and scheduler commands against spam.
  - `moderation.py`: Structured JSON audit listeners (`on_ready`, `on_message`, `on_command_error`) converting missing arguments, invalid inputs, cooldown errors, 404s, upstream failures, and unexpected exceptions into clean red Discord Embeds.
- **Zero Hardcoded Secrets & Fail-Fast Startup (`bot/config.py`)**:
  - Reads `DISCORD_TOKEN` strictly from environment variables / `.env`.
  - Rejects empty, malformed, or placeholder tokens immediately before touching the Discord gateway.
- **Offline Mock-Gateway Verification**:
  - Complete async test suite (`tests/test_bot.py`) exercising every command, Embed structure, `httpx.MockTransport` HTTP flow, SQLite persistence, cooldown enforcement, and error handling - 100% offline with zero live token required.

## Project Structure

```text
08_discord_bot/
├── bot/
│   ├── __init__.py
│   ├── config.py          # Env config loader + fail-fast token validation
│   ├── logging_utils.py   # Single-line structured JSON log formatter
│   ├── rate_limit.py      # Per-(command, user) cooldown guard
│   ├── storage.py         # SQLite WAL ReminderStore
│   ├── services.py        # Async OpenMeteoClient, GitHubClient, Embed builders
│   ├── main.py            # ModularDiscordBot entrypoint + graceful shutdown + offline demo
│   └── cogs/
│       ├── general.py     # !hello, /ping, /status
│       ├── automation.py  # /weather, /github, /remind
│       └── moderation.py  # on_ready, on_message, on_command_error
├── tests/
│   └── test_bot.py        # 14 async mock-gateway pytest tests
├── .env.example
├── Dockerfile
├── pytest.ini
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Offline Verification & Mock-Gateway Demo (No Discord Token Required)

```bash
# Run full async test suite (14 tests)
pytest -v

# Run end-to-end offline mock-gateway demo
python -m bot.main --demo-offline
```

### 3. Configure Discord Developer Portal (Live Bot Setup)

1. Open the [Discord Developer Portal](https://discord.com/developers/applications) and click **New Application**.
2. Go to **Bot** -> **Add Bot** -> **Reset Token** and copy your bot token.
3. Under **Privileged Gateway Intents**, enable **Message Content Intent** (required for prefix commands like `!hello`).
4. Go to **OAuth2** -> **URL Generator**:
   - Scopes: `bot`, `applications.commands`
   - Bot Permissions: `Send Messages`, `Embed Links`, `Read Message History`
   - Open the generated invite URL to add the bot to your test server.
5. Copy `.env.example` to `.env` and set `DISCORD_TOKEN`:

```bash
cp .env.example .env
# Edit .env and set DISCORD_TOKEN=<your_real_token>
```

6. Validate config fail-fast and launch:

```bash
python -m bot.main --check
python -m bot.main
```

## 24/7 Docker Deployment

```bash
docker build -t modular-discord-bot .
docker run -d \
  --name discord-bot \
  --restart unless-stopped \
  --env-file .env \
  -v bot_data:/app/data \
  modular-discord-bot
```

## Verification Summary

- **Pytest Suite**: `14/14 PASS` (offline async mock-gateway + `httpx.MockTransport` + SQLite temp DB + fail-fast CLI check).
- **Offline Demo**: Verified `!hello`, `/ping`, `/status` (live RSS memory reporting), `/weather Tokyo`, `/github python/cpython`, `/remind` SQLite persistence + async delivery, and cooldown rejection Embed via `ModerationCog.on_command_error`.

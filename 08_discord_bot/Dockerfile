FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    REMINDER_DB_PATH=/app/data/reminders.db

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY bot/ ./bot/

RUN mkdir -p /app/data && useradd --create-home --uid 10001 botuser && chown -R botuser:botuser /app
USER botuser

HEALTHCHECK --interval=60s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import sqlite3, os; sqlite3.connect(os.getenv('REMINDER_DB_PATH', '/app/data/reminders.db')).close()" || exit 1

CMD ["python", "-m", "bot.main"]

"""Minimal RFC-4180 CSV parser (Benchmark 3: quoted comma & escaped quote bug)."""

from __future__ import annotations


def parse_csv_line(line: str) -> list[str]:
    """Parse a single RFC-4180 CSV line into fields."""
    if line == "":
        return []
    return [cell.strip('"') for cell in line.split(",")]


def parse_csv(text: str) -> list[list[str]]:
    """Parse multi-line CSV text into rows of string fields."""
    rows: list[list[str]] = []
    for raw_line in text.splitlines():
        if raw_line == "":
            continue
        rows.append(parse_csv_line(raw_line))
    return rows

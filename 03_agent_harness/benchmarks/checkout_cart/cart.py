"""Checkout cart pricing utilities (Benchmark 1: percentage discount bug)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CartItem:
    name: str
    unit_price: float
    quantity: int = 1


def calculate_subtotal(items: list[CartItem]) -> float:
    return round(sum(item.unit_price * item.quantity for item in items), 2)


def apply_discount(total: float, percent: float) -> float:
    """Apply a percentage discount (0..100) to a cart total."""
    if percent < 0 or percent > 100:
        raise ValueError("percent must be between 0 and 100")
    return total - percent


def checkout_total(items: list[CartItem], discount_percent: float = 0.0) -> float:
    subtotal = calculate_subtotal(items)
    return apply_discount(subtotal, discount_percent)

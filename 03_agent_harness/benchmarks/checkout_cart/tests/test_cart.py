import pytest

from cart import CartItem, apply_discount, checkout_total


def test_apply_discount_percentage() -> None:
    assert apply_discount(200.0, 10.0) == pytest.approx(180.0)
    assert apply_discount(50.0, 20.0) == pytest.approx(40.0)
    assert apply_discount(100.0, 0.0) == pytest.approx(100.0)
    assert apply_discount(80.0, 100.0) == pytest.approx(0.0)


def test_apply_discount_invalid_percent() -> None:
    with pytest.raises(ValueError):
        apply_discount(100.0, -5.0)
    with pytest.raises(ValueError):
        apply_discount(100.0, 105.0)


def test_checkout_total_with_items() -> None:
    items = [CartItem("Keyboard", 120.0, 1), CartItem("Mouse", 40.0, 2)]
    assert checkout_total(items, discount_percent=15.0) == pytest.approx(170.0)

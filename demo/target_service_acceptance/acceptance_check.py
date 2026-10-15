"""Hidden acceptance checks for the Lab 1 target service repair."""

from __future__ import annotations

import os
import sys
from decimal import Decimal as D
from pathlib import Path


repo = Path(os.environ["HARNESS_REPO"]).resolve()
service = repo / "target-service"
sys.path.insert(0, str(service))

from orders.pricing import quote  # noqa: E402
from orders.service import place_order  # noqa: E402
from orders.storage import OrderStore  # noqa: E402


def lines(price: str, quantity: int = 1) -> list[dict[str, object]]:
    return [{"unit_price": D(price), "quantity": quantity}]


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


check(quote(lines("50"), D("0.10"))["total"] == D("50.00"), "shipping was discounted")

threshold = quote(lines("100"), D("0.10"))
check(threshold["shipping"] == D("5.00"), "shipping threshold used pre-discount goods")
check(threshold["total"] == D("95.00"), "discounted threshold total is wrong")

check(quote(lines("10.005"))["total"] == D("15.01"), "ROUND_HALF_UP was not used")

per_line = quote(lines("0.335") + lines("0.335"))
check(per_line["subtotal"] == D("0.68"), "line totals were not rounded before summing")
check(per_line["total"] == D("5.68"), "rounded line total is wrong")

check(quote(lines("10.05"), D("0.10"))["total"] == D("14.04"), "discount amount was not rounded")
check(quote(lines("125"), D("0.20"))["total"] == D("100.00"), "exact free-shipping threshold failed")
check(quote(lines("20", 2))["total"] == D("45.00"), "no-discount behavior regressed")

invalid_inputs = [
    ([], D("0")),
    (lines("-1"), D("0")),
    (lines("10", 0), D("0")),
    (lines("10", True), D("0")),
    (lines("10"), D("0.31")),
    (lines("NaN"), D("0")),
]
for items, discount in invalid_inputs:
    try:
        quote(items, discount)
    except ValueError:
        pass
    else:
        raise AssertionError(f"invalid input was accepted: {items!r}, {discount!r}")

store = OrderStore()
expected = place_order(store, "O-001", lines("20"))
check(store.get("O-001") == expected, "order was not persisted")
expected["total"] = D("999")
check(store.get("O-001")["total"] == D("25.00"), "stored order was mutated through returned data")
try:
    place_order(store, "O-001", lines("20"))
except ValueError:
    pass
else:
    raise AssertionError("duplicate order was accepted")

print("target-service acceptance checks passed")

"""Mock tool implementations for the banking chatbot."""

from __future__ import annotations

from datetime import datetime
from typing import Dict
from uuid import uuid4


def calculate_emi(P: float, annual_rate_percent: float, months: int) -> float:
    """Compute the Equated Monthly Instalment."""

    if months <= 0:
        raise ValueError("Months must be positive")

    monthly_rate = annual_rate_percent / 12 / 100
    if monthly_rate == 0:
        return P / months

    numerator = P * monthly_rate * (1 + monthly_rate) ** months
    denominator = (1 + monthly_rate) ** months - 1
    return numerator / denominator


def block_card(card_number: str, reason: str) -> str:
    """Simulate blocking a card and return a ticket id."""

    _ = card_number  # The number is used to perform the mock action only.
    reference = f"BLK-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{uuid4().hex[:6].upper()}"
    return reference


def fetch_rate(product: str) -> Dict[str, str | float]:
    """Return mock rate information for banking products."""

    catalog = {
        "savings account": 3.0,
        "fixed deposit": 6.5,
        "loan": 9.25,
        "home loan": 8.1,
        "credit card": 42.0,
    }
    rate = catalog.get(product.lower(), 0.0)
    return {
        "tool": "fetch_rate",
        "product": product,
        "rate_percent": rate,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

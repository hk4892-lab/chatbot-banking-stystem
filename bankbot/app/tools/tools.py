"""Typed tool implementations used by the policy router."""
from __future__ import annotations

import math
import uuid
from typing import Any, Dict

from app.core.redaction import restore_tokens


def calculate_emi(
    P: float,
    annual_rate_percent: float,
    months: int,
    token_map: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    """Compute the monthly EMI for a loan."""

    if months <= 0:
        raise ValueError("months must be positive")

    monthly_rate = annual_rate_percent / 12 / 100
    if monthly_rate == 0:
        emi = P / months
    else:
        emi = P * monthly_rate * math.pow(1 + monthly_rate, months) / (
            math.pow(1 + monthly_rate, months) - 1
        )

    emi_value = round(emi, 2)
    summary = (
        f"Estimated EMI is ?{emi_value:,.2f} for principal ?{P:,.2f} at {annual_rate_percent:.2f}% annual rate"
        f" over {months} months."
    )
    return {
        "tool": "calculate_emi",
        "emi": emi_value,
        "summary": summary,
        "parameters": {
            "principal": P,
            "annual_rate_percent": annual_rate_percent,
            "months": months,
        },
    }


def block_card(
    tokenized_card: str,
    reason: str,
    token_map: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    """Mock a card block operation and return a ticket identifier."""

    token_map = token_map or {}
    resolved_card = restore_tokens(tokenized_card, token_map) if tokenized_card else None
    # We never return the resolved card number, but we ensure the token can be processed.
    _ = resolved_card  # placeholder for potential integration
    ticket_id = f"BLK-{uuid.uuid4().hex[:10].upper()}"
    summary = "Card block request submitted. Ticket ID: {ticket_id}.".format(ticket_id=ticket_id)

    return {
        "tool": "block_card",
        "ticket_id": ticket_id,
        "card_token": tokenized_card,
        "reason": reason,
        "summary": summary,
    }


def fetch_rate(
    product: str,
    token_map: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    """Return a mock rate sheet for supported banking products."""

    product = product.lower()
    rates = {
        "savings": {"interest_percent": 3.5, "notes": "Savings account interest paid quarterly."},
        "fd": {"interest_percent": 6.25, "notes": "Fixed deposit rate for 1-year tenure."},
        "fixed deposit": {"interest_percent": 6.25, "notes": "Fixed deposit rate for 1-year tenure."},
        "home loan": {"interest_percent": 8.5, "notes": "Floating rate linked to repo."},
        "education loan": {"interest_percent": 9.1, "notes": "Concession available for women borrowers."},
        "personal loan": {"interest_percent": 12.5, "notes": "Indicative rate; subject to credit score."},
    }

    info = rates.get(product, {"interest_percent": None, "notes": "Product not found."})
    summary = (
        f"Current indicative rate for {product.title()}: {info['interest_percent']}%"
        if info["interest_percent"] is not None
        else f"I could not find a rate for {product.title()}."
    )

    return {
        "tool": "fetch_rate",
        "product": product,
        "data": info,
        "summary": summary,
    }

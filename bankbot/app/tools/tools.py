"""Mock banking tools used by the chatbot."""

from __future__ import annotations

import hashlib
import math
from typing import Dict


def emi_calculator(principal: float, annual_rate: float, n_months: int) -> Dict[str, float]:
    """Compute a simple EMI schedule."""

    principal = max(principal, 0.0)
    annual_rate = max(annual_rate, 0.0)
    n_months = max(int(n_months), 1)

    monthly_rate = annual_rate / (12 * 100)
    if monthly_rate == 0:
        emi_value = principal / n_months
    else:
        factor = math.pow(1 + monthly_rate, n_months)
        emi_value = principal * monthly_rate * factor / (factor - 1)

    total_payment = emi_value * n_months
    total_interest = total_payment - principal

    return {
        "emi": round(emi_value, 2),
        "total_interest": round(total_interest, 2),
        "total_payment": round(total_payment, 2),
    }


def block_card_ticket(name: str, last4: str, reason: str) -> Dict[str, str]:
    sanitized_last4 = (last4 or "0000")[-4:]
    digest_source = f"{name.lower()}|{sanitized_last4}|{reason.lower()}".encode("utf-8")
    digest = hashlib.sha1(digest_source).hexdigest()[:6].upper()
    ticket_id = f"TKT-{sanitized_last4}-{digest}"
    return {"ticket_id": ticket_id, "status": "raised"}


_RATE_TABLE = {
    "savings": 3.5,
    "fixed_deposit": 6.7,
    "home_loan": 8.4,
    "personal_loan": 12.5,
}


def interest_rates(product: str) -> Dict[str, float]:
    key = (product or "").strip().lower().replace(" ", "_")
    rate = _RATE_TABLE.get(key, 4.0)
    return {"product": key or "general", "rate_percent": rate}


__all__ = ["emi_calculator", "block_card_ticket", "interest_rates"]

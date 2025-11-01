"""Evaluate BankBot against a small multilingual prompt set."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd
from fastapi.testclient import TestClient

from app.api import app


def run_eval(prompts_path: Path, use_sentence_transformers: bool = False) -> Dict[str, float]:
    client = TestClient(app)
    df = pd.read_csv(prompts_path)

    total = 0
    correct = 0
    refusals = 0
    confidences: List[float] = []

    for _, row in df.iterrows():
        prompt = str(row["prompt"]).strip()
        expected = str(row["expected_contains"]).strip().lower()
        if not prompt:
            continue

        response = client.post(
            "/chat/turn",
            json={
                "message": prompt,
                "lang": "AUTO",
                "confidence_threshold": 0.18,
                "use_sentence_transformers": use_sentence_transformers,
            },
        )
        response.raise_for_status()
        data = response.json()
        answer = (data.get("answer") or "").lower()
        route = data.get("route")
        confidence = float(data.get("confidence", 0.0))

        total += 1
        if expected and expected in answer:
            correct += 1
        if route == "escalate":
            refusals += 1
        confidences.append(confidence)

    return {
        "total": total,
        "accuracy": (correct / total) if total else 0.0,
        "refusal_rate": (refusals / total) if total else 0.0,
        "avg_confidence": (sum(confidences) / len(confidences)) if confidences else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BankBot prompts")
    parser.add_argument(
        "--prompts",
        type=Path,
        default=Path(__file__).resolve().with_name("prompts.csv"),
        help="CSV file with prompt,expected_contains",
    )
    parser.add_argument(
        "--use-sentence-transformers",
        action="store_true",
        help="Enable optional sentence-transformers reranker during evaluation",
    )
    args = parser.parse_args()

    metrics = run_eval(args.prompts, use_sentence_transformers=args.use_sentence_transformers)
    print("Evaluation summary")
    print("-------------------")
    print(f"Prompts: {metrics['total']}")
    print(f"Top-1 accuracy: {metrics['accuracy']:.2%}")
    print(f"Refusal rate: {metrics['refusal_rate']:.2%}")
    print(f"Average confidence: {metrics['avg_confidence']:.3f}")


if __name__ == "__main__":
    main()

"""Simple evaluation for the banking chatbot retrieval."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from app.core.retrieval import RetrievalEngine

DATA_DIR = Path(__file__).resolve().parent.parent / "app" / "data"
DEFAULT_THRESHOLD = 0.18


@dataclass
class EvalSample:
    prompt: str
    expected_contains: str
    lang: str


def load_samples(path: Path) -> list[EvalSample]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        samples: list[EvalSample] = []
        for row in reader:
            samples.append(
                EvalSample(
                    prompt=row["prompt"].strip(),
                    expected_contains=row["expected_contains"].strip(),
                    lang=row.get("lang", "EN").strip() or "EN",
                )
            )
    return samples


def evaluate(samples: Iterable[EvalSample], use_sentence_transformers: bool) -> None:
    retriever = RetrievalEngine(DATA_DIR, use_sentence_transformers=use_sentence_transformers)
    total = 0
    correct = 0
    refusals = 0
    cumulative_score = 0.0

    print(f"Using sentence transformers: {use_sentence_transformers}")
    print("prompt,top_id,top_score,match")

    for sample in samples:
        total += 1
        results = retriever.search(sample.prompt, top_k=1)
        if results:
            top = results[0]
            top_score = top.score
            top_content = top.entry.content.lower()
            match = sample.expected_contains.lower() in top_content
            if match:
                correct += 1
            cumulative_score += top_score
            if top_score < DEFAULT_THRESHOLD:
                refusals += 1
            print(f"{sample.prompt},{top.entry.id},{top_score:.3f},{match}")
        else:
            refusals += 1
            print(f"{sample.prompt},NONE,0.000,False")

    accuracy = correct / total if total else 0.0
    refusal_rate = refusals / total if total else 0.0
    avg_score = cumulative_score / total if total else 0.0

    print("\nSummary")
    print(f"Samples: {total}")
    print(f"Top-1 accuracy: {accuracy:.2%}")
    print(f"Refusal rate (< {DEFAULT_THRESHOLD}): {refusal_rate:.2%}")
    print(f"Average top score: {avg_score:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BankBot retrieval on a CSV of prompts")
    parser.add_argument("csv", type=Path, help="CSV file with prompt and expected_contains columns")
    parser.add_argument(
        "--use-sentence-transformers",
        action="store_true",
        help="Enable sentence-transformer re-ranking if installed",
    )
    args = parser.parse_args()

    samples = load_samples(args.csv)
    evaluate(samples, use_sentence_transformers=args.use_sentence_transformers)


if __name__ == "__main__":
    main()

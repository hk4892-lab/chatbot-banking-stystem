"""Evaluation harness for grounded response quality."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Dict, List

from app.deps import get_pipeline


CITATION_REGEX = re.compile(r'\[[^\]]+\([^\]]+\)\]')


def load_prompts(path: Path) -> List[Dict[str, str]]:
    with path.open('r', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def compute_metrics(rows: List[Dict[str, str]], k: int) -> Dict[str, float]:
    pipeline = get_pipeline()
    total = len(rows)
    hits = 0
    ndcg_sum = 0.0
    grounded_sentences = 0
    total_sentences = 0

    for row in rows:
        query = row['query']
        expected = row['expected_doc_id']
        result = pipeline.answer(query, k=k, lang='AUTO', temperature=0.0)
        context_ids = result['used_context_ids']
        if expected in context_ids:
            hits += 1
            rank = context_ids.index(expected)
            ndcg_sum += 1.0 / math.log2(rank + 2)
        reply = result['reply']
        sentences = [segment.strip() for segment in re.split(r'[.!?]\s+', reply) if segment.strip()]
        total_sentences += len(sentences)
        grounded_sentences += sum(1 for sentence in sentences if CITATION_REGEX.search(sentence))

    recall = hits / total if total else 0.0
    ndcg = ndcg_sum / total if total else 0.0
    grounding = grounded_sentences / total_sentences if total_sentences else 0.0
    return {
        'recall@k': recall,
        'ndcg@k': ndcg,
        'grounded_sentence_ratio': grounding,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate the chatbot grounding metrics.')
    parser.add_argument('--prompts', type=Path, default=Path('eval/prompts.sample.csv'))
    parser.add_argument('--k', type=int, default=5)
    args = parser.parse_args()
    rows = load_prompts(args.prompts)
    metrics = compute_metrics(rows, args.k)
    for key, value in metrics.items():
        print(f'{key}: {value:.4f}')


if __name__ == '__main__':  # pragma: no cover
    main()

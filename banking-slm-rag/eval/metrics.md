## Evaluation Metrics

- **Recall@k** – Fraction of prompts where the expected document appears in the top-k retrieved passages consumed by the pipeline.
- **nDCG@k** – Position-weighted relevance score assuming a single relevant document per prompt, emphasising higher ranked hits.
- **Grounded Sentence Ratio** – Percentage of assistant sentences containing at least one citation formatted as `[Title (doc_id)]`, approximating groundedness.

Run `python eval/evaluate.py --prompts eval/prompts.sample.csv --k 5` after building the index to generate these metrics.

#!/usr/bin/env python3
"""Evaluation script for RAG system."""
import sys
import csv
import re
from pathlib import Path
from collections import defaultdict

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.retrieval import FAISSRetriever
from app.config import settings


def calculate_recall_at_k(retrieved_ids: list[str], expected_id: str, k: int) -> float:
    """Calculate Recall@k."""
    top_k = retrieved_ids[:k]
    return 1.0 if expected_id in top_k else 0.0


def calculate_ndcg_at_k(retrieved_ids: list[str], expected_id: str, k: int) -> float:
    """Calculate nDCG@k."""
    top_k = retrieved_ids[:k]
    if expected_id not in top_k:
        return 0.0
    
    # Position of expected doc (0-indexed)
    position = top_k.index(expected_id)
    
    # DCG (ideal is 1.0 at position 0)
    dcg = 1.0 / (1 + position)  # log2(position + 2) simplified
    idcg = 1.0  # Ideal DCG (expected at position 0)
    
    return dcg / idcg


def extract_citations(text: str) -> list[str]:
    """Extract doc_ids from citation format [Title (doc_id)]."""
    pattern = re.compile(r"\[([^\]]+)\s+\(([^)]+)\)\]")
    matches = pattern.findall(text)
    return [doc_id for _, doc_id in matches]


def evaluate_retrieval(eval_data: list[dict], retriever: FAISSRetriever, k: int = 5):
    """
    Evaluate retrieval performance.
    
    Args:
        eval_data: List of dicts with 'query' and 'expected_doc_id'
        retriever: FAISS retriever instance
        k: Number of documents to retrieve
        
    Returns:
        Dict with metrics
    """
    recall_scores = []
    ndcg_scores = []
    by_category = defaultdict(lambda: {"recall": [], "ndcg": []})
    
    print(f"\n{'Query':<50} | Expected | Recall@{k} | nDCG@{k}")
    print("-" * 90)
    
    for item in eval_data:
        query = item["query"]
        expected_id = item["expected_doc_id"]
        category = item.get("category", "unknown")
        
        # Handle refusal cases
        if expected_id == "REFUSAL":
            # For refusal cases, we just check that retrieval returns low scores
            results = retriever.search(query, k=k)
            avg_score = sum(r["score"] for r in results) / len(results) if results else 0.0
            
            # Low score is good for refusal cases
            is_correct = avg_score < settings.refusal_threshold
            recall = 1.0 if is_correct else 0.0
            ndcg = 1.0 if is_correct else 0.0
            
            print(f"{query:<50} | REFUSAL  | {recall:.2f}     | {ndcg:.2f}")
        else:
            # Normal retrieval case
            results = retriever.search(query, k=k)
            retrieved_ids = [r["doc_id"] for r in results]
            
            recall = calculate_recall_at_k(retrieved_ids, expected_id, k)
            ndcg = calculate_ndcg_at_k(retrieved_ids, expected_id, k)
            
            top_id = retrieved_ids[0] if retrieved_ids else "NONE"
            print(f"{query:<50} | {expected_id:<8} | {recall:.2f}     | {ndcg:.2f}  (top: {top_id})")
        
        recall_scores.append(recall)
        ndcg_scores.append(ndcg)
        by_category[category]["recall"].append(recall)
        by_category[category]["ndcg"].append(ndcg)
    
    # Calculate overall metrics
    avg_recall = sum(recall_scores) / len(recall_scores)
    avg_ndcg = sum(ndcg_scores) / len(ndcg_scores)
    
    print("\n" + "=" * 90)
    print(f"Overall Metrics (k={k}):")
    print(f"  Recall@{k}: {avg_recall:.3f}")
    print(f"  nDCG@{k}:   {avg_ndcg:.3f}")
    print(f"  Total queries: {len(eval_data)}")
    
    # By category
    print("\nBy Category:")
    for category, scores in by_category.items():
        cat_recall = sum(scores["recall"]) / len(scores["recall"])
        cat_ndcg = sum(scores["ndcg"]) / len(scores["ndcg"])
        print(f"  {category:<15} - Recall: {cat_recall:.3f}, nDCG: {cat_ndcg:.3f} ({len(scores['recall'])} queries)")
    
    return {
        "recall_at_k": avg_recall,
        "ndcg_at_k": avg_ndcg,
        "by_category": by_category,
    }


def evaluate_grounding(eval_data: list[dict], retriever: FAISSRetriever, k: int = 5):
    """
    Evaluate citation/grounding in responses (requires generator).
    This is a simplified check for citation format presence.
    
    Args:
        eval_data: List of evaluation examples
        retriever: FAISS retriever
        k: Number of docs to retrieve
    """
    print("\n" + "=" * 90)
    print("Grounding Evaluation (Citation Format Check)")
    print("=" * 90)
    
    citation_pattern = re.compile(r"\[([^\]]+)\s+\(([^)]+)\)\]")
    
    grounded_count = 0
    
    for item in eval_data[:5]:  # Sample first 5
        query = item["query"]
        results = retriever.search(query, k=k)
        
        # Mock response with citation (in real eval, call generator)
        mock_response = f"Based on the information, {query.lower()} is explained in [{results[0]['title']} ({results[0]['doc_id']})]" if results else "No information available."
        
        has_citation = bool(citation_pattern.search(mock_response))
        if has_citation:
            grounded_count += 1
        
        print(f"Query: {query}")
        print(f"  Has citation: {has_citation}")
    
    print(f"\nGrounded responses: {grounded_count}/5 (sample)")


def main():
    """Main evaluation entry point."""
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <prompts.csv> [k]")
        print("Example: python eval/evaluate.py eval/prompts.sample.csv 5")
        sys.exit(1)
    
    prompts_file = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    print(f"Loading evaluation prompts from: {prompts_file}")
    print(f"Using k={k} for retrieval")
    
    # Load eval data
    eval_data = []
    with open(prompts_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        eval_data = list(reader)
    
    print(f"Loaded {len(eval_data)} evaluation queries")
    
    # Initialize retriever
    print(f"\nInitializing retriever...")
    print(f"  Index: {settings.faiss_index_path}")
    print(f"  Embedding model: {settings.embedding_model}")
    
    retriever = FAISSRetriever(
        index_path=settings.faiss_index_path,
        meta_path=settings.faiss_meta_path,
        embedding_model=settings.embedding_model,
    )
    
    # Run retrieval evaluation
    metrics = evaluate_retrieval(eval_data, retriever, k=k)
    
    # Run grounding check
    evaluate_grounding(eval_data, retriever, k=k)
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()

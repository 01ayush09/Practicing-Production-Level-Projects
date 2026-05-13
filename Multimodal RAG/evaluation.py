"""
evaluation.py — RAGAS evaluation for the RAG pipeline.

Metrics computed:
  - Faithfulness:       Is the answer grounded in the retrieved context?
  - Answer Relevancy:   Does the answer actually address the question?
  - Context Precision:  Are the retrieved chunks relevant to the question?

These three metrics together give a measurable, reproducible quality
score for the RAG system — transforming this from a demo into a
proper ML project with quantitative benchmarking.
"""

import os
from typing import List, Dict, Any


def evaluate_rag_response(
    question: str,
    answer: str,
    contexts: List[str],
    ground_truth: str = None,
) -> Dict[str, Any]:
    """
    Evaluate a single RAG response using Gemini-based RAGAS-style metrics.

    Args:
        question:     The user's question
        answer:       The generated answer
        contexts:     List of retrieved context strings
        ground_truth: Optional reference answer for comparison

    Returns:
        Dictionary with scores for each metric (0.0 - 1.0)
    """
    import google.generativeai as genai
    from dotenv import load_dotenv

    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set in .env")
    genai.configure(api_key=api_key)

    model = genai.GenerativeModel("gemini-2.0-flash")
    scores = {}

    # ── Metric 1: Faithfulness ────────────────────────────────────────────────
    # Does every claim in the answer have support in the retrieved context?
    context_text = "\n\n".join(contexts)
    faithfulness_prompt = f"""You are evaluating a RAG system's faithfulness.

QUESTION: {question}

RETRIEVED CONTEXT:
{context_text}

GENERATED ANSWER:
{answer}

TASK: Score how faithful the answer is to the retrieved context.
- 1.0 = Every claim in the answer is directly supported by the context
- 0.5 = Most claims are supported, some are hallucinated
- 0.0 = Answer contains many claims not found in the context

Respond with ONLY a JSON object like this (no explanation, no markdown):
{{"score": 0.85, "reason": "Brief reason in one sentence"}}"""

    try:
        resp = model.generate_content(faithfulness_prompt)
        import json, re
        raw = resp.text.strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        result = json.loads(raw)
        scores["faithfulness"] = {
            "score": float(result.get("score", 0.0)),
            "reason": result.get("reason", ""),
        }
    except Exception as e:
        scores["faithfulness"] = {"score": 0.0, "reason": f"Evaluation error: {e}"}

    # ── Metric 2: Answer Relevancy ────────────────────────────────────────────
    # Does the answer actually address what was asked?
    relevancy_prompt = f"""You are evaluating a RAG system's answer relevancy.

QUESTION: {question}

GENERATED ANSWER:
{answer}

TASK: Score how relevant and complete the answer is to the question.
- 1.0 = Answer directly and completely addresses the question
- 0.5 = Answer is partially relevant but misses key aspects
- 0.0 = Answer does not address the question at all

Respond with ONLY a JSON object like this (no explanation, no markdown):
{{"score": 0.85, "reason": "Brief reason in one sentence"}}"""

    try:
        resp = model.generate_content(relevancy_prompt)
        raw = resp.text.strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        result = json.loads(raw)
        scores["answer_relevancy"] = {
            "score": float(result.get("score", 0.0)),
            "reason": result.get("reason", ""),
        }
    except Exception as e:
        scores["answer_relevancy"] = {"score": 0.0, "reason": f"Evaluation error: {e}"}

    # ── Metric 3: Context Precision ───────────────────────────────────────────
    # Are the retrieved chunks actually useful for answering the question?
    precision_prompt = f"""You are evaluating a RAG system's context precision.

QUESTION: {question}

RETRIEVED CONTEXTS:
{context_text}

TASK: Score how precise and relevant the retrieved contexts are for answering the question.
- 1.0 = All retrieved contexts are highly relevant to the question
- 0.5 = Some contexts are relevant, others are noise
- 0.0 = Retrieved contexts are mostly irrelevant to the question

Respond with ONLY a JSON object like this (no explanation, no markdown):
{{"score": 0.85, "reason": "Brief reason in one sentence"}}"""

    try:
        resp = model.generate_content(precision_prompt)
        raw = resp.text.strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        result = json.loads(raw)
        scores["context_precision"] = {
            "score": float(result.get("score", 0.0)),
            "reason": result.get("reason", ""),
        }
    except Exception as e:
        scores["context_precision"] = {"score": 0.0, "reason": f"Evaluation error: {e}"}

    # ── Overall Score ─────────────────────────────────────────────────────────
    metric_scores = [scores[m]["score"] for m in scores]
    scores["overall"] = round(sum(metric_scores) / len(metric_scores), 3)

    return scores


def format_scores_for_display(scores: Dict[str, Any]) -> str:
    """Format evaluation scores into a readable string for the UI."""
    if not scores:
        return ""

    lines = ["\n\n---\n### 📊 RAGAS Evaluation Scores\n"]

    metrics = {
        "faithfulness": "🔍 Faithfulness",
        "answer_relevancy": "🎯 Answer Relevancy",
        "context_precision": "📌 Context Precision",
    }

    for key, label in metrics.items():
        if key in scores:
            score = scores[key]["score"]
            reason = scores[key]["reason"]
            bar = _score_bar(score)
            lines.append(f"**{label}**: {score:.2f} {bar}")
            lines.append(f"  _{reason}_\n")

    overall = scores.get("overall", 0)
    lines.append(f"**⭐ Overall Score**: {overall:.2f} / 1.00")

    return "\n".join(lines)


def _score_bar(score: float) -> str:
    """Visual bar for score display."""
    filled = int(score * 10)
    empty = 10 - filled
    return f"[{'█' * filled}{'░' * empty}]"


def run_batch_evaluation(test_cases: List[Dict]) -> List[Dict]:
    """
    Run RAGAS evaluation on multiple test cases.

    Args:
        test_cases: List of dicts with keys: question, answer, contexts, ground_truth (optional)

    Returns:
        List of dicts with original data + scores

    Example:
        test_cases = [
            {
                "question": "What is RAG?",
                "answer": "RAG stands for...",
                "contexts": ["RAG is a technique...", "Retrieval augmented..."],
            }
        ]
    """
    results = []
    for i, case in enumerate(test_cases):
        print(f"Evaluating case {i+1}/{len(test_cases)}: {case['question'][:50]}...")
        scores = evaluate_rag_response(
            question=case["question"],
            answer=case["answer"],
            contexts=case.get("contexts", []),
            ground_truth=case.get("ground_truth"),
        )
        results.append({**case, "scores": scores})
        print(f"  Overall: {scores['overall']:.2f}")
    return results


if __name__ == "__main__":
    # Quick test
    scores = evaluate_rag_response(
        question="What is RAG?",
        answer="RAG stands for Retrieval-Augmented Generation. It combines retrieval of relevant documents with language model generation to produce grounded answers.",
        contexts=[
            "Retrieval-Augmented Generation (RAG) is a technique that enhances LLMs by retrieving relevant documents.",
            "RAG systems consist of a retriever and a generator component working together.",
        ],
    )
    print(format_scores_for_display(scores))

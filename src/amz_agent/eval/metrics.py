from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable


def classification_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    labels = sorted(set(y_true) | set(y_pred))
    total = len(y_true) or 1
    accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / total
    per_label: dict[str, dict[str, float]] = {}
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(y_true, y_pred))
        fp = sum(t != label and p == label for t, p in zip(y_true, y_pred))
        fn = sum(t == label and p != label for t, p in zip(y_true, y_pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": sum(t == label for t in y_true),
        }
    confusion = defaultdict(int)
    for true, pred in zip(y_true, y_pred):
        confusion[f"{true}->{pred}"] += 1
    return {
        "accuracy": round(accuracy, 4),
        "per_label": per_label,
        "confusion": dict(confusion),
    }


def retrieval_metrics(
    cases: list[dict],
    rankings: dict[str, list[str]],
    k: int,
    sufficient: dict[str, bool],
    top_texts: dict[str, str] | None = None,
) -> dict:
    recalls: list[float] = []
    precisions: list[float] = []
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []
    no_answer_correct: list[float] = []
    heading_hits: list[float] = []

    for case in cases:
        expected = set(case.get("expected_doc_ids", []))
        ranked = rankings.get(case["id"], [])[:k]
        if not case.get("answerable", True):
            no_answer_correct.append(0.0 if sufficient.get(case["id"], True) else 1.0)
            continue
        hits = [1.0 if doc in expected else 0.0 for doc in ranked]
        recalls.append(1.0 if any(hits) else 0.0)
        precisions.append(sum(hits) / len(ranked) if ranked else 0.0)
        first = next((i + 1 for i, hit in enumerate(hits) if hit), None)
        reciprocal_ranks.append(1.0 / first if first else 0.0)
        dcg = sum(hit / math.log2(i + 2) for i, hit in enumerate(hits))
        ideal_hits = min(len(expected), k)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
        ndcgs.append(dcg / idcg if idcg else 0.0)
        if case.get("expected_heading_keywords"):
            top_text = (top_texts or {}).get(case["id"], " ".join(ranked[:1]))
            heading_hits.append(
                1.0
                if any(keyword.lower() in top_text.lower() for keyword in case["expected_heading_keywords"])
                else 0.0
            )

    def mean(values: Iterable[float]) -> float:
        values = list(values)
        return round(sum(values) / len(values), 4) if values else 0.0

    return {
        "recall_at_k": mean(recalls),
        "precision_at_k": mean(precisions),
        "mrr_at_k": mean(reciprocal_ranks),
        "ndcg_at_k": mean(ndcgs),
        "no_answer_accuracy": mean(no_answer_correct),
        "heading_hit_at_1": mean(heading_hits),
        "evaluated": len(recalls),
        "no_answer_cases": len(no_answer_correct),
    }


def agent_summary(rows: list[dict]) -> dict:
    if not rows:
        return {"cases": 0}
    total = len(rows)
    return {
        "cases": total,
        "decision_accuracy": round(sum(r["decision_ok"] for r in rows) / total, 4),
        "handoff_accuracy": round(sum(r["handoff_ok"] for r in rows) / total, 4),
        "tool_recall": round(sum(r["tool_recall"] for r in rows) / total, 4),
        "citation_rate": round(sum(r["has_citation"] for r in rows) / total, 4),
        "guardrail_pass_rate": round(sum(r["guardrail_ok"] for r in rows) / total, 4),
        "task_success": round(sum(r["success"] for r in rows) / total, 4),
        "avg_steps": round(sum(r["steps"] for r in rows) / total, 2),
        "avg_latency_ms": round(sum(r["latency_ms"] for r in rows) / total, 2),
    }

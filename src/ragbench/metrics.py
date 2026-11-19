"""Standard IR metrics (same definitions as BEIR / trec_eval)."""
from __future__ import annotations

import math

Run = dict[str, list[str]]            # query_id -> ranked doc_ids
Qrels = dict[str, dict[str, int]]     # query_id -> {doc_id: relevance}


def ndcg_at_k(ranked: list[str], rels: dict[str, int], k: int = 10) -> float:
    dcg = sum((2 ** rels.get(d, 0) - 1) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted(rels.values(), reverse=True)[:k]
    idcg = sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked: list[str], rels: dict[str, int], k: int) -> float:
    relevant = {d for d, r in rels.items() if r > 0}
    return len(relevant & set(ranked[:k])) / len(relevant) if relevant else 0.0


def mrr_at_k(ranked: list[str], rels: dict[str, int], k: int = 10) -> float:
    for i, d in enumerate(ranked[:k]):
        if rels.get(d, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


def evaluate(run: Run, qrels: Qrels, ks=(1, 5, 10, 100)) -> dict[str, float]:
    qids = [q for q in qrels if q in run]
    out = {"ndcg@10": sum(ndcg_at_k(run[q], qrels[q], 10) for q in qids) / len(qids),
           "mrr@10": sum(mrr_at_k(run[q], qrels[q], 10) for q in qids) / len(qids)}
    for k in ks:
        out[f"recall@{k}"] = sum(recall_at_k(run[q], qrels[q], k) for q in qids) / len(qids)
    return {k: round(v, 4) for k, v in out.items()}


def per_query(run: Run, qrels: Qrels, k: int = 10) -> dict[str, float]:
    return {q: ndcg_at_k(run[q], qrels[q], k) for q in qrels if q in run}

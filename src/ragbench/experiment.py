"""Benchmark retrieval strategies on BEIR SciFact.

Usage: python -m ragbench.experiment
"""
from __future__ import annotations

import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import data
from .metrics import evaluate, per_query, recall_at_k
from .retrievers import (
    BM25Retriever,
    CrossEncoderReranker,
    DenseRetriever,
    HybridRetriever,
    SentenceTransformerEncoder,
)

REPORTS = data.ROOT / "reports"
FIGS = REPORTS / "figures"
DENSE_MODELS = {"Dense: MiniLM-L6": "sentence-transformers/all-MiniLM-L6-v2",
                "Dense: BGE-small": "BAAI/bge-small-en-v1.5"}
CURVE_KS = [1, 3, 5, 10, 20, 50, 100]


def paired_bootstrap(a: dict[str, float], b: dict[str, float], n: int = 2000, seed: int = 0):
    """95% CI for mean(b - a) over queries, plus P(diff <= 0)."""
    qids = sorted(a)
    d = np.array([b[q] - a[q] for q in qids])
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def build_retrievers(ds: data.Dataset, encoders: dict | None = None, reranker_scorer=None):
    ids = list(ds.corpus)
    texts = [ds.doc_text(i) for i in ids]
    lookup = dict(zip(ids, texts, strict=True))
    timings, retrievers = {}, []

    t = time.perf_counter()
    bm25 = BM25Retriever(ids, texts)
    timings["BM25"] = time.perf_counter() - t
    retrievers.append(bm25)

    encoders = encoders or {name: SentenceTransformerEncoder(m) for name, m in DENSE_MODELS.items()}
    dense = {}
    for name, enc in encoders.items():
        t = time.perf_counter()
        dense[name] = DenseRetriever(name, enc, ids, texts)
        timings[name] = time.perf_counter() - t
        retrievers.append(dense[name])

    best_dense = dense[list(dense)[-1]]
    hybrid = HybridRetriever(f"Hybrid RRF (BM25 + {best_dense.name.split(': ')[-1]})", [bm25, best_dense])
    retrievers.append(hybrid)
    retrievers.append(CrossEncoderReranker(f"{hybrid.name} + cross-encoder rerank", hybrid, lookup,
                                           top_n=50, scorer=reranker_scorer))
    return retrievers, timings, lookup


def run(ds: data.Dataset, retrievers, index_timings: dict) -> tuple[pd.DataFrame, dict]:
    rows, runs = [], {}
    for r in retrievers:
        t = time.perf_counter()
        runs[r.name] = r.search(ds.queries, k=100)
        ms = 1000 * (time.perf_counter() - t) / len(ds.queries)
        m = evaluate(runs[r.name], ds.qrels)
        rows.append({"retriever": r.name, **m, "latency_ms_per_query": round(ms, 2),
                     "index_seconds": round(index_timings.get(r.name, 0.0), 1)})
        print(f"{r.name:55s} nDCG@10={m['ndcg@10']:.4f} R@100={m['recall@100']:.4f} {ms:.1f} ms/q", flush=True)
    return pd.DataFrame(rows), runs


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    ds = data.load()
    print(f"corpus={len(ds.corpus):,} queries={len(ds.queries)}", flush=True)
    retrievers, timings, _ = build_retrievers(ds)
    results, runs = run(ds, retrievers, timings)

    names = list(runs)
    pq = {n: per_query(runs[n], ds.qrels) for n in names}
    base, best = "BM25", results.sort_values("ndcg@10").iloc[-1]["retriever"]
    dense_name = names[2]
    diff, lo, hi = paired_bootstrap(pq[base], pq[best])
    wins = {"dense_better": sum(pq[dense_name][q] > pq[base][q] + 1e-9 for q in pq[base]),
            "bm25_better": sum(pq[base][q] > pq[dense_name][q] + 1e-9 for q in pq[base]),
            "tie": sum(abs(pq[base][q] - pq[dense_name][q]) <= 1e-9 for q in pq[base])}
    summary = {"n_docs": len(ds.corpus), "n_queries": len(ds.queries), "best": best,
               "best_vs_bm25_ndcg10": {"mean_diff": round(diff, 4), "ci95": [round(lo, 4), round(hi, 4)]},
               "per_query_bm25_vs_dense": {"dense": dense_name, **wins},
               "results": results.to_dict(orient="records")}
    (REPORTS / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    (REPORTS / "results.md").write_text(results.to_markdown(index=False) + "\n")

    # ---- figures ----
    plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})
    short = [n.replace(" + cross-encoder rerank", "\n+ rerank").replace("Hybrid RRF ", "Hybrid\n") for n in names]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, metric, color in zip(axes, ["ndcg@10", "recall@100"], ["#2563eb", "#16a34a"], strict=True):
        vals = results[metric].to_numpy()
        ax.bar(range(len(vals)), vals, color=color)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
        ax.set_xticks(range(len(vals)), short, fontsize=7)
        ax.set(title=metric, ylim=(0, vals.max() * 1.15))
    fig.tight_layout()
    fig.savefig(FIGS / "retriever_comparison.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for n in names:
        curve = [np.mean([recall_at_k(runs[n][q], ds.qrels[q], k) for q in ds.qrels]) for k in CURVE_KS]
        ax.plot(CURVE_KS, curve, marker="o", label=n, lw=1.5)
    ax.set(xscale="log", xlabel="k (documents retrieved)", ylabel="Recall@k",
           title="How many documents must the LLM read?")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGS / "recall_curve.png")

    fig, ax = plt.subplots(figsize=(5.5, 5))
    xs = np.array([pq[base][q] for q in pq[base]])
    ys = np.array([pq[dense_name][q] for q in pq[base]])
    jitter = np.random.default_rng(0).normal(0, 0.01, (2, len(xs)))
    ax.scatter(xs + jitter[0], ys + jitter[1], s=10, alpha=0.5, color="#7c3aed")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set(xlabel="BM25 nDCG@10", ylabel=f"{dense_name} nDCG@10",
           title=f"Per-query: dense wins {wins['dense_better']}, BM25 wins {wins['bm25_better']}")
    fig.tight_layout()
    fig.savefig(FIGS / "per_query_scatter.png")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()

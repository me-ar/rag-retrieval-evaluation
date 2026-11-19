"""Offline tests: no model downloads (hashing encoder + stub cross-encoder)."""
import math

import pytest
from fastapi.testclient import TestClient

from ragbench import experiment
from ragbench.data import Dataset
from ragbench.metrics import evaluate, mrr_at_k, ndcg_at_k, recall_at_k
from ragbench.rag import answer, build_context
from ragbench.retrievers import (
    BM25Retriever,
    CrossEncoderReranker,
    DenseRetriever,
    HashingEncoder,
    reciprocal_rank_fusion,
    tokenize,
)

DOCS = {
    "d1": {"title": "Vitamin D", "text": "Vitamin D supplementation reduces fracture risk in elderly adults."},
    "d2": {"title": "Statins", "text": "Statins lower LDL cholesterol and cardiovascular events."},
    "d3": {"title": "Sleep", "text": "Short sleep duration is associated with obesity in children."},
    "d4": {"title": "Exercise", "text": "Aerobic exercise improves insulin sensitivity in type 2 diabetes."},
}
DS = Dataset(DOCS, {"q1": "does vitamin d reduce fractures", "q2": "statins and cholesterol"},
             {"q1": {"d1": 1}, "q2": {"d2": 1}})


def test_metrics_known_values():
    assert ndcg_at_k(["a", "b"], {"a": 1}) == 1.0
    assert ndcg_at_k(["b", "a"], {"a": 1}) == pytest.approx(1 / math.log2(3))
    assert recall_at_k(["a", "x"], {"a": 1, "b": 1}, 2) == 0.5
    assert mrr_at_k(["x", "y", "a"], {"a": 1}) == pytest.approx(1 / 3)
    assert ndcg_at_k(["x"], {}) == 0.0


def test_tokenize_drops_stopwords_keeps_hyphens():
    assert tokenize("The IL-6 levels of the patients") == ["il-6", "levels", "patients"]


def test_bm25_ranks_relevant_doc_first():
    ids = list(DOCS)
    r = BM25Retriever(ids, [DS.doc_text(i) for i in ids])
    run = r.search(DS.queries, k=4)
    assert run["q1"][0] == "d1" and run["q2"][0] == "d2"
    assert evaluate(run, DS.qrels)["mrr@10"] == 1.0


def test_dense_retriever_with_hashing_encoder():
    ids = list(DOCS)
    r = DenseRetriever("hash", HashingEncoder(), ids, [DS.doc_text(i) for i in ids])
    assert r.search(DS.queries, k=2)["q2"][0] == "d2"


def test_rrf_rewards_agreement():
    fused = reciprocal_rank_fusion([{"q": ["a", "b", "c"]}, {"q": ["b", "a", "d"]}], k=4)
    assert set(fused["q"][:2]) == {"a", "b"} and fused["q"][-1] in {"c", "d"}


def test_reranker_reorders_head_only():
    class Fixed:
        name = "fixed"

        def search(self, queries, k):
            return {q: ["d3", "d2", "d1", "d4"][:k] for q in queries}

    rr = CrossEncoderReranker("rr", Fixed(), {d: DS.doc_text(d) for d in DOCS}, top_n=2,
                              scorer=lambda pairs: [len(p[1]) for p in pairs])
    out = rr.search({"q": "x"}, k=4)["q"]
    assert out[2:] == ["d1", "d4"] and set(out[:2]) == {"d2", "d3"}


def test_full_benchmark_pipeline_offline():
    stub = lambda pairs: [len(set(tokenize(q)) & set(tokenize(d))) for q, d in pairs]  # noqa: E731
    retrievers, timings, _ = experiment.build_retrievers(DS, encoders={"Dense: hash": HashingEncoder()},
                                                         reranker_scorer=stub)
    results, runs = experiment.run(DS, retrievers, timings)
    assert len(results) == 4
    assert (results["ndcg@10"] > 0.5).all()


def test_rag_answer_extractive_cites_sources(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    lookup = {d: DS.doc_text(d) for d in DOCS}
    out = answer("vitamin d fracture risk", ["d1", "d2"], lookup, k=2)
    assert out["mode"] == "extractive" and "d1" in out["citations"]
    assert build_context(["d1"], lookup).startswith("[d1]")


def test_api(monkeypatch):
    monkeypatch.setenv("RAG_ENCODER", "hashing")
    from api import main
    main.STATE["index"] = main.build_index(DS)
    with TestClient(main.app) as c:
        assert c.get("/health").json()["documents"] == 4
        r = c.get("/search", params={"q": "statins cholesterol", "k": 2, "method": "bm25"}).json()
        assert r["results"][0]["doc_id"] == "d2"
        assert c.get("/search", params={"q": "x"}).status_code == 422
        assert "d1" in c.get("/answer", params={"q": "vitamin d fractures"}).json()["citations"]

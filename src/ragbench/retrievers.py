"""Retrievers: sparse (BM25), dense (bi-encoder), hybrid (reciprocal rank fusion) and a
cross-encoder re-ranking stage. All share one interface: `search(queries, k) -> {qid: [doc_ids]}`."""
from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from typing import Protocol

import numpy as np
from rank_bm25 import BM25Okapi

STOPWORDS = set("""a an and are as at be by for from has have in is it its of on or that the this to was
were will with not no than then there these those which who whom what when where why how""".split())
TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOPWORDS]


class Retriever(Protocol):
    name: str

    def search(self, queries: dict[str, str], k: int = 100) -> dict[str, list[str]]: ...


# ---------------------------------------------------------------- sparse
class BM25Retriever:
    def __init__(self, doc_ids: Sequence[str], texts: Sequence[str], k1: float = 0.9, b: float = 0.4):
        self.name = "BM25"
        self.doc_ids = list(doc_ids)
        self.bm25 = BM25Okapi([tokenize(t) for t in texts], k1=k1, b=b)

    def scores(self, query: str) -> np.ndarray:
        return self.bm25.get_scores(tokenize(query))

    def search(self, queries, k=100):
        out = {}
        for qid, q in queries.items():
            s = self.scores(q)
            top = np.argsort(-s)[:k]
            out[qid] = [self.doc_ids[i] for i in top]
        return out


# ---------------------------------------------------------------- dense
class Encoder(Protocol):
    def encode_queries(self, texts: list[str]) -> np.ndarray: ...
    def encode_docs(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerEncoder:
    """Wraps a sentence-transformers model; handles model-specific query instructions."""

    QUERY_PREFIX = {"BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
                    "intfloat/e5-small-v2": "query: "}
    DOC_PREFIX = {"intfloat/e5-small-v2": "passage: "}

    def __init__(self, model_name: str, batch_size: int = 64):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device="cpu")
        self.batch_size = batch_size

    def _enc(self, texts, prefix):
        return self.model.encode([prefix + t for t in texts], batch_size=self.batch_size,
                                 normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)

    def encode_queries(self, texts):
        return self._enc(texts, self.QUERY_PREFIX.get(self.model_name, ""))

    def encode_docs(self, texts):
        return self._enc(texts, self.DOC_PREFIX.get(self.model_name, ""))


class HashingEncoder:
    """Dependency-free bag-of-words hashing encoder — used in tests and as a sanity baseline."""

    def __init__(self, dim: int = 2048):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for t in tokenize(text):
            v[int(hashlib.md5(t.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode_queries(self, texts):
        return np.stack([self._vec(t) for t in texts])

    encode_docs = encode_queries


class DenseRetriever:
    def __init__(self, name: str, encoder: Encoder, doc_ids: Sequence[str], texts: Sequence[str]):
        self.name = name
        self.encoder = encoder
        self.doc_ids = list(doc_ids)
        self.doc_emb = encoder.encode_docs(list(texts)).astype(np.float32)

    def search(self, queries, k=100):
        qids = list(queries)
        q = self.encoder.encode_queries([queries[i] for i in qids]).astype(np.float32)
        sims = q @ self.doc_emb.T          # cosine similarity (embeddings are L2-normalised)
        k = min(k, sims.shape[1])
        top = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        out = {}
        for row, qid in enumerate(qids):
            idx = top[row][np.argsort(-sims[row, top[row]])]
            out[qid] = [self.doc_ids[i] for i in idx]
        return out


# ---------------------------------------------------------------- hybrid + rerank
def reciprocal_rank_fusion(runs: list[dict[str, list[str]]], k: int = 100, c: int = 60) -> dict[str, list[str]]:
    """RRF (Cormack et al., 2009): score(d) = Σ 1/(c + rank). Robust, needs no score calibration."""
    fused = {}
    for qid in runs[0]:
        scores: dict[str, float] = {}
        for run in runs:
            for rank, d in enumerate(run.get(qid, [])):
                scores[d] = scores.get(d, 0.0) + 1.0 / (c + rank + 1)
        fused[qid] = sorted(scores, key=scores.get, reverse=True)[:k]
    return fused


class HybridRetriever:
    def __init__(self, name: str, retrievers: list, depth: int = 100):
        self.name, self.retrievers, self.depth = name, retrievers, depth

    def search(self, queries, k=100):
        return reciprocal_rank_fusion([r.search(queries, self.depth) for r in self.retrievers], k=k)


class CrossEncoderReranker:
    """Re-scores the top-N candidates of a first-stage retriever with a cross-encoder."""

    def __init__(self, name: str, first_stage, doc_lookup: dict[str, str], top_n: int = 50,
                 model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", scorer=None):
        self.name, self.first_stage, self.docs, self.top_n = name, first_stage, doc_lookup, top_n
        if scorer is None:
            from sentence_transformers import CrossEncoder
            model = CrossEncoder(model_name, device="cpu")
            scorer = lambda pairs: model.predict(pairs, batch_size=64, show_progress_bar=False)  # noqa: E731
        self.scorer = scorer

    def search(self, queries, k=100):
        first = self.first_stage.search(queries, max(k, self.top_n))
        out = {}
        for qid, cands in first.items():
            head, tail = cands[: self.top_n], cands[self.top_n:]
            s = np.asarray(self.scorer([(queries[qid], self.docs[d]) for d in head]))
            out[qid] = ([head[i] for i in np.argsort(-s)] + tail)[:k]
        return out

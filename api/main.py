"""Search + answer API over the SciFact corpus.

Run: uvicorn api.main:app        Docs: http://localhost:8000/docs
Set RAG_ENCODER=hashing for a dependency-free demo (no model download).
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Query

from ragbench import data
from ragbench.rag import answer
from ragbench.retrievers import (
    BM25Retriever,
    DenseRetriever,
    HashingEncoder,
    HybridRetriever,
    SentenceTransformerEncoder,
)

STATE: dict = {}


def build_index(ds: data.Dataset) -> dict:
    ids = list(ds.corpus)
    texts = [ds.doc_text(i) for i in ids]
    enc = (HashingEncoder() if os.getenv("RAG_ENCODER") == "hashing"
           else SentenceTransformerEncoder("BAAI/bge-small-en-v1.5"))
    bm25 = BM25Retriever(ids, texts)
    dense = DenseRetriever("dense", enc, ids, texts)
    return {"bm25": bm25, "dense": dense, "hybrid": HybridRetriever("hybrid", [bm25, dense]),
            "lookup": dict(zip(ids, texts, strict=True)), "corpus": ds.corpus}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if "index" not in STATE:
        STATE["index"] = build_index(data.load())
    yield


app = FastAPI(title="SciFact RAG API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "documents": len(STATE["index"]["lookup"])}


@app.get("/search")
def search(q: str = Query(min_length=3, max_length=500), k: int = Query(5, ge=1, le=50),
           method: Literal["bm25", "dense", "hybrid"] = "hybrid"):
    idx = STATE["index"]
    ids = idx[method].search({"q": q}, k=k)["q"]
    return {"query": q, "method": method,
            "results": [{"doc_id": d, "title": idx["corpus"][d]["title"],
                         "snippet": idx["lookup"][d][:300]} for d in ids]}


@app.get("/answer")
def ask(q: str = Query(min_length=3, max_length=500), k: int = Query(5, ge=1, le=20)):
    idx = STATE["index"]
    ids = idx["hybrid"].search({"q": q}, k=k)["q"]
    if not ids:
        raise HTTPException(404, "No documents found")
    return answer(q, ids, idx["lookup"], k=k)

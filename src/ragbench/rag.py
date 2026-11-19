"""Answer generation on top of retrieval, with inline citations.

If ANTHROPIC_API_KEY is set, an LLM writes a grounded answer citing [doc_id]s; otherwise an
extractive fallback returns the most query-relevant sentences, so the pipeline always works.
"""
from __future__ import annotations

import os
import re

from .retrievers import tokenize

PROMPT = """Answer the question using ONLY the sources below. Cite every claim with the source id
in square brackets, e.g. [12345]. If the sources do not contain the answer, say so.

Sources:
{context}

Question: {question}
Answer:"""


def build_context(doc_ids: list[str], doc_lookup: dict[str, str], max_chars: int = 6000) -> str:
    parts, used = [], 0
    for d in doc_ids:
        chunk = f"[{d}] {doc_lookup[d]}"
        if used + len(chunk) > max_chars:
            break
        parts.append(chunk)
        used += len(chunk)
    return "\n\n".join(parts)


def extractive_answer(question: str, doc_ids: list[str], doc_lookup: dict[str, str], n: int = 2) -> str:
    q = set(tokenize(question))
    scored = []
    for d in doc_ids:
        for sent in re.split(r"(?<=[.!?])\s+", doc_lookup[d]):
            overlap = len(q & set(tokenize(sent))) / (len(q) or 1)
            scored.append((overlap, sent.strip(), d))
    scored.sort(key=lambda x: -x[0])
    return " ".join(f"{s} [{d}]" for _, s, d in scored[:n])


def answer(question: str, doc_ids: list[str], doc_lookup: dict[str, str], k: int = 5) -> dict:
    top = doc_ids[:k]
    if os.getenv("ANTHROPIC_API_KEY"):
        import anthropic
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=os.getenv("RAG_MODEL", "claude-haiku-5-5"), max_tokens=400,
            messages=[{"role": "user", "content": PROMPT.format(
                context=build_context(top, doc_lookup), question=question)}])
        text, mode = msg.content[0].text, "llm"
    else:
        text, mode = extractive_answer(question, top, doc_lookup), "extractive"
    cited = sorted(set(re.findall(r"\[([^\]]+)\]", text)) & set(top))
    return {"answer": text, "citations": cited, "sources": top, "mode": mode}

"""BEIR SciFact: 5,183 scientific abstracts, 300 test claims with graded relevance judgements."""
from __future__ import annotations

import csv
import io
import json
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"


@dataclass
class Dataset:
    corpus: dict[str, dict]          # doc_id -> {"title", "text"}
    queries: dict[str, str]          # query_id -> text
    qrels: dict[str, dict[str, int]]  # query_id -> {doc_id: relevance}

    def doc_text(self, doc_id: str) -> str:
        d = self.corpus[doc_id]
        return f"{d['title']}. {d['text']}".strip(". ")


def download() -> Path:
    target = DATA_DIR / "scifact"
    if (target / "corpus.jsonl").exists():
        return target
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading BEIR SciFact...")
    with urllib.request.urlopen(URL, timeout=120) as r:
        zipfile.ZipFile(io.BytesIO(r.read())).extractall(DATA_DIR)
    return target


def _read_jsonl(path: Path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def load(split: str = "test", path: Path | None = None) -> Dataset:
    path = path or download()
    corpus = {d["_id"]: {"title": d.get("title", ""), "text": d["text"]}
              for d in _read_jsonl(path / "corpus.jsonl")}
    qrels: dict[str, dict[str, int]] = {}
    with open(path / "qrels" / f"{split}.tsv", encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])
    queries = {q["_id"]: q["text"] for q in _read_jsonl(path / "queries.jsonl") if q["_id"] in qrels}
    return Dataset(corpus, queries, qrels)

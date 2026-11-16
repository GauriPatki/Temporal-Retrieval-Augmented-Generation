"""
Loads ChroniclingAmericaQA, builds a search index, and retrieves passages.

Reuses the team's re-ranker from src/mrag_integration.py (topic/year split,
sliding-window MaxSim, temporal decay score). Uses the public
facebook/contriever-msmarco model, so no fine-tuned weights or GPU are needed.
"""

import hashlib
import os
import random
import re
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # torch + faiss on macOS

import faiss
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import nltk  # noqa: E402

for _pkg in ("punkt", "punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{_pkg}")
    except LookupError:
        nltk.download(_pkg, quiet=True)

import mrag_integration as mrag  # noqa: E402  (the team's re-ranker)

MODEL_NAME = "facebook/contriever-msmarco"
CACHE_DIR = Path(__file__).parent / "cache"
YEAR_RE = re.compile(r"\b(18[0-9]{2}|19[0-9]{2}|20[0-2][0-9])\b")


def _pick(ex, keys):
    """Return the first non-empty field among `keys` as a string."""
    for key in keys:
        value = ex.get(key)
        if isinstance(value, dict) and "text" in value:
            value = value["text"]
        if isinstance(value, (list, tuple)):
            value = value[0] if value else None
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def load_caqa(n_questions=50, corpus_size=2000, seed=42, pool=50):
    """
    Returns (passages, questions, dates).

    questions: year-explicit CAQA questions that have a gold answer.
    passages:  gold passages of a fixed pool of `pool` questions plus random
               other CAQA passages as distractors, up to corpus_size (kept small
               so it runs on CPU). The corpus depends only on `pool`, not on
               n_questions, so running fewer questions reuses the cached index.
    """
    ds = load_dataset("Bhawna/ChroniclingAmericaQA", split="validation")
    print(f"Loaded ChroniclingAmericaQA validation: {len(ds)} rows, columns={ds.column_names}")

    rows, dates_by_text = [], {}
    for ex in ds:
        q = _pick(ex, ["question", "query"])
        p = _pick(ex, ["context", "positive_passage", "passage"])
        a = _pick(ex, ["answer", "answers", "answer_text", "gold_answer"])
        if q and p:
            rows.append((q, p, a))
            date = _pick(ex, ["publication_date", "date"])
            if date and p not in dates_by_text:
                dates_by_text[p] = date

    year_rows = [r for r in rows if YEAR_RE.search(r[0]) and r[2]]
    if not year_rows:
        raise ValueError("No year-explicit questions with answers found. Check the column names printed above.")

    rng = random.Random(seed)
    rng.shuffle(year_rows)
    pool_rows = year_rows[:max(pool, n_questions)]
    eval_rows = pool_rows[:n_questions]

    passages, seen = [], set()
    for _, p, _ in pool_rows:
        if p not in seen:
            seen.add(p)
            passages.append(p)

    others = list(dict.fromkeys(r[1] for r in rows if r[1] not in seen))
    rng.shuffle(others)
    passages.extend(others[: max(0, corpus_size - len(passages))])

    pid = {p: i for i, p in enumerate(passages)}
    questions = [{"question": q, "answer": a, "gold_id": pid[p]} for q, p, a in eval_rows]
    print(f"Eval questions: {len(questions)} | Corpus passages: {len(passages)}")
    dates = [dates_by_text.get(p) for p in passages]  # newspaper publication date, like a record timestamp
    return passages, questions, dates


def _with_publication_year(date, years):
    """Put the publication year first, so the date checks use the record's own timestamp."""
    match = re.search(r"\b(1[5-9]\d{2}|20\d{2})\b", str(date or ""))
    if not match:
        return list(years)
    pub = int(match.group(1))
    return [pub] + [y for y in years if y != pub]


class Retriever:
    """Dense search with optional time-aware re-ranking (the team's MRAG step)."""

    def __init__(self, passages, dates=None):
        self.passages = passages
        self.dates = dates or [None] * len(passages)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME).to(mrag.DEVICE).eval()

        CACHE_DIR.mkdir(exist_ok=True)
        key = hashlib.md5("".join(p[:50] for p in passages).encode()).hexdigest()[:10]
        emb_path = CACHE_DIR / f"passages_{len(passages)}_{key}.npy"
        win_path = CACHE_DIR / f"windows_{len(passages)}_{key}.pt"

        if emb_path.exists():
            embs = np.load(emb_path)
        else:
            print("Encoding passages (first run only, cached afterwards)...")
            embs = mrag.encode_texts(self.model, self.tokenizer, passages)
            np.save(emb_path, embs)

        self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(embs.shape[1]))
        self.index.add_with_ids(embs, np.arange(len(passages), dtype=np.int64))

        if win_path.exists():
            saved = torch.load(win_path, weights_only=False)
            self.win_tensor, self.win_map = saved["tensor"], saved["map"]
        else:
            print("Encoding sentence windows for the re-ranker (first run only)...")
            self.win_tensor, self.win_map = mrag.precompute_window_embeddings(
                self.model, self.tokenizer, passages
            )
            torch.save({"tensor": self.win_tensor, "map": self.win_map}, win_path)

    def search(self, question, k=3, rerank=True, pool=100):
        """
        rerank=False: baseline, the full question goes straight to dense search.
        rerank=True:  the team's pipeline, search with the topic part of the
                      question, then re-rank with the full question (year score).
        """
        query = mrag.extract_mc(question) if rerank else question
        q_emb = mrag.encode_texts(self.model, self.tokenizer, [query])
        scores, ids = self.index.search(q_emb, pool)
        cand = [int(i) for i in ids[0] if i >= 0]

        if rerank:
            ranked = mrag.mrag_rerank_1(
                question,
                [self.passages[c] for c in cand],
                cand,
                self.model,
                self.tokenizer,
                base_scores=scores[0][: len(cand)],
                blend_weight=0.0,
                temporal_weight=1.0,
                window_emb_tensor=self.win_tensor,
                doc_window_map=self.win_map,
            )
        else:
            ranked = cand

        return [
            {
                "id": pid,
                "published": self.dates[pid],
                "years": _with_publication_year(self.dates[pid], mrag.get_doc_years(pid, self.passages[pid])),
                "text": self.passages[pid],
            }
            for pid in ranked[:k]
        ]

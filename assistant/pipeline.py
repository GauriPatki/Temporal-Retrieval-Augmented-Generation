"""Retrieve -> Claude answers -> checks -> one retry -> passed or needs review."""

import json
import time
from pathlib import Path

from checks import check_answer
from llm import ask_claude

RUNS_DIR = Path(__file__).parent / "runs"


def answer_question(retriever, question, rerank=True, retry=True, k=3, log_file=None):
    retrieved = retriever.search(question, k=k, rerank=rerank)

    raw = ask_claude(question, retrieved)
    first, first_failures = check_answer(question, raw, retrieved)
    final, final_failures, attempts = first, first_failures, 1

    if retry and first_failures:
        raw = ask_claude(question, retrieved, feedback=first_failures)
        final, final_failures = check_answer(question, raw, retrieved)
        attempts = 2

    record = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "question": question,
        "setup": ("time_aware" if rerank else "baseline") + ("+check" if retry else ""),
        "retrieved": [{"id": p["id"], "years": p["years"][:10]} for p in retrieved],
        "first_answer": first,
        "first_failures": first_failures,
        "final_answer": final,
        "final_failures": final_failures,
        "attempts": attempts,
        "status": "passed" if not final_failures else "needs_review",
    }

    if log_file:
        RUNS_DIR.mkdir(exist_ok=True)
        with open(RUNS_DIR / log_file, "a") as f:
            f.write(json.dumps(record) + "\n")

    return record, retrieved

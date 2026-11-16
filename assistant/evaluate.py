"""
Compares three setups on year-explicit ChroniclingAmericaQA questions:

  A. Baseline retrieval + Claude
  B. Time-aware retrieval + Claude            (first answer, no check)
  C. Time-aware retrieval + Claude + checks   (one retry, then flag for review)

B and C share Claude's first call, so the cost is about 2-3 calls per question.

Run from the repo root:
  python assistant/evaluate.py --n 50
  python assistant/evaluate.py --n 50 --resume   # continue after Ctrl+C
"""

import argparse
import json
import time
from pathlib import Path

import llm
from pipeline import answer_question
from scoring import is_correct, is_not_found
from retrieval import Retriever, load_caqa

RESULTS_DIR = Path(__file__).parent / "results"
PROGRESS = RESULTS_DIR / "progress.jsonl"


def _answer_text(parsed):
    if not isinstance(parsed, dict) or parsed.get("answer") is None:
        return ""
    return str(parsed["answer"])


def _has(failures, code):
    return any(f.startswith(code) for f in failures)


def run_one(retriever, q, log_file):
    """Runs one question through setup A and setups B/C. Returns a result row."""
    base, base_ret = answer_question(retriever, q["question"], rerank=False, retry=False, log_file=log_file)
    ta, ta_ret = answer_question(retriever, q["question"], rerank=True, retry=True, log_file=log_file)

    row = {
        "question": q["question"],
        "gold": q["answer"],
        "A_hit": q["gold_id"] in [p["id"] for p in base_ret],
        "A_answer": _answer_text(base["first_answer"]),
        "A_failures": base["first_failures"],
        "BC_hit": q["gold_id"] in [p["id"] for p in ta_ret],
        "B_answer": _answer_text(ta["first_answer"]),
        "B_failures": ta["first_failures"],
        "C_answer": _answer_text(ta["final_answer"]),
        "C_failures": ta["final_failures"],
        "C_flagged": ta["status"] == "needs_review",
    }
    for s in "ABC":
        row[f"{s}_correct"] = is_correct(row[f"{s}_answer"], q["answer"])
    return row


def write_results(rows, n_passages):
    """Writes results.md, per_question.json and showcase.json. Returns the table text."""
    n = len(rows)
    if n == 0:
        return None
    pct = lambda count: f"{100 * count / n:.0f}%"  # noqa: E731
    count = lambda fn: sum(1 for r in rows if fn(r))  # noqa: E731

    # Each answer ends up as exactly one of: correct, "not found" (left for a person),
    # flagged by the check (sent for review), or a wrong answer shown to a person.
    def outcomes(s):
        answered = lambda r: not is_not_found(r[f"{s}_answer"])  # noqa: E731
        flagged = (lambda r: r["C_flagged"]) if s == "C" else (lambda r: False)
        hit = "A_hit" if s == "A" else "BC_hit"
        return {
            "gold_in_top3": count(lambda r: r[hit]),
            "correct": count(lambda r: r[f"{s}_correct"]),
            "not_found": count(lambda r: not answered(r) and not r[f"{s}_correct"]),
            "flagged": count(lambda r: flagged(r)) if s == "C" else None,
            "wrong_shown": count(lambda r: answered(r) and not r[f"{s}_correct"] and not flagged(r)),
        }

    summary = {s: outcomes(s) for s in "ABC"}
    caught = count(lambda r: r["C_flagged"] and not r["C_correct"])
    false_alarms = count(lambda r: r["C_flagged"] and r["C_correct"])
    wrong_time = {s: count(lambda r, s=s: _has(r[f"{s}_failures"], "WRONG_TIME")) for s in "ABC"}

    model = llm.model_label()
    names = {
        "A": f"A. Baseline retrieval + {model}",
        "B": f"B. Time-aware retrieval + {model}",
        "C": f"C. Time-aware + {model} + checks",
    }
    lines = [
        f"Results on {n} year-explicit ChroniclingAmericaQA questions ({n_passages}-passage index)",
        "",
        "| Setup | Gold passage in top 3 | Correct answer | Said \"not found\" | Flagged for review | Wrong answer shown to a person |",
        "|---|---|---|---|---|---|",
    ]
    for s in "ABC":
        m = summary[s]
        lines.append(
            f"| {names[s]} | {pct(m['gold_in_top3'])} | {pct(m['correct'])} | {pct(m['not_found'])} | "
            f"{'—' if m['flagged'] is None else pct(m['flagged'])} | {pct(m['wrong_shown'])} |"
        )
    lines += [
        "",
        f"The check flagged {summary['C']['flagged']} answers. Caught (actually wrong): {caught}. False alarms (actually correct): {false_alarms}.",
        f"Answers citing a wrong-year source: A={wrong_time['A']}, B={wrong_time['B']}, C={wrong_time['C']}.",
        "Correct = loose match against the dataset's answer key (contains, or token F1 >= 0.5).",
    ]
    table = "\n".join(lines)

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "results.md").write_text(table + "\n", encoding="utf-8")
    (RESULTS_DIR / "per_question.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    showcase = {
        "time_aware_fixed_it": [r["question"] for r in rows if not r["A_correct"] and r["C_correct"]][:10],
        "check_caught_bad_answer": [r["question"] for r in rows if r["C_flagged"] and not r["C_correct"]][:10],
        "baseline_cited_wrong_year": [r["question"] for r in rows if _has(r["A_failures"], "WRONG_TIME")][:10],
    }
    (RESULTS_DIR / "showcase.json").write_text(json.dumps(showcase, indent=2), encoding="utf-8")
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50, help="number of questions")
    parser.add_argument("--corpus", type=int, default=2000, help="number of passages in the index")
    parser.add_argument("--resume", action="store_true", help="continue a run that was stopped")
    args = parser.parse_args()

    passages, questions, dates = load_caqa(n_questions=args.n, corpus_size=args.corpus)
    retriever = Retriever(passages, dates)
    log_file = f"eval_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"

    RESULTS_DIR.mkdir(exist_ok=True)
    rows = []
    if args.resume and PROGRESS.exists():
        rows = [json.loads(line) for line in PROGRESS.read_text(encoding="utf-8").splitlines() if line.strip()]
        for row in rows:  # re-score with the current scoring code
            for s in "ABC":
                row[f"{s}_correct"] = is_correct(row[f"{s}_answer"], row["gold"])
        print(f"Resuming: {len(rows)} questions already done")
    else:
        PROGRESS.write_text("", encoding="utf-8")
    done = {r["question"] for r in rows}

    for i, q in enumerate(questions, 1):
        if q["question"] in done:
            continue
        started = time.time()
        try:
            row = run_one(retriever, q, log_file)
        except Exception as err:  # one bad model reply should not stop a long run
            print(f"[{i}/{len(questions)}] SKIPPED because of an error: {err!r}  {q['question'][:60]}")
            continue
        rows.append(row)
        with open(PROGRESS, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        write_results(rows, len(passages))  # saved after every question, so Ctrl+C loses nothing
        print(f"[{i}/{len(questions)}] A={'✓' if row['A_correct'] else '✗'} "
              f"B={'✓' if row['B_correct'] else '✗'} C={'✓' if row['C_correct'] else '✗'}"
              f"{' (flagged)' if row['C_flagged'] else ''}  ({time.time() - started:.0f}s)  {q['question'][:55]}")

    table = write_results(rows, len(passages))
    if table is None:
        raise SystemExit("No questions finished. See the errors above.")
    print("\n" + table)
    print(f"\nSaved: {RESULTS_DIR / 'results.md'}, per_question.json, showcase.json")
    print(f"Every LLM call is logged in: assistant/runs/{log_file}")


if __name__ == "__main__":
    main()

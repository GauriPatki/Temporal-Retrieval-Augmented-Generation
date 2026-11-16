"""
Side-by-side demo for the screen recording.

Run from the repo root:
  python assistant/demo.py --showcase            # first question where time-aware fixed the baseline
  python assistant/demo.py --showcase caught     # first question where the check caught a bad answer
  python assistant/demo.py "Who was the mayor of Richmond in 1912?"

Use the same --n/--corpus as evaluate.py so the cached index is reused.
"""

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import llm
from pipeline import answer_question
from retrieval import Retriever, load_caqa

console = Console()
SHOWCASE = Path(__file__).parent / "results" / "showcase.json"
KINDS = {"fixed": "time_aware_fixed_it", "caught": "check_caught_bad_answer", "wrongyear": "baseline_cited_wrong_year"}


def show(title, record, retrieved, gold_id=None):
    table = Table(show_header=True, header_style="bold", expand=True)
    table.add_column("Rank", width=4)
    table.add_column("Passage", width=8)
    table.add_column("Published / years", width=24)
    table.add_column("Text")
    for rank, p in enumerate(retrieved, 1):
        mark = " ★" if p["id"] == gold_id else ""
        when = p.get("published") or ", ".join(map(str, p["years"][:6])) or "—"
        table.add_row(str(rank), f"{p['id']}{mark}", str(when),
                      p["text"][:110].replace("\n", " ").rstrip(". ") + "…")

    answer = record["final_answer"] or {}
    lines = [f"[bold]{llm.model_label()} answer:[/bold] {answer.get('answer', '(unreadable output)')}",
             f"[bold]Cites passages:[/bold] {answer.get('cited_ids', [])}"]
    if record["attempts"] == 2:
        lines.append("[yellow]First answer failed the checks, so it was retried once:[/yellow]")
        lines += [f"   [yellow]✗ {f}[/yellow]" for f in record["first_failures"]]
    if record["status"] == "passed":
        lines.append("[bold green]✓ Passed all checks[/bold green]")
    else:
        lines.append("[bold red]⚠ Flagged: needs human review[/bold red]")
        lines += [f"   [red]✗ {f}[/red]" for f in record["final_failures"]]

    console.print(Panel.fit(title, style="bold cyan"))
    console.print(table)
    console.print("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?")
    parser.add_argument("--showcase", nargs="?", const="fixed", choices=list(KINDS))
    parser.add_argument("--pick", type=int, default=0, help="which showcase example to use (0, 1, 2...)")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--corpus", type=int, default=2000)
    args = parser.parse_args()

    passages, questions, dates = load_caqa(n_questions=args.n, corpus_size=args.corpus)
    if args.showcase:
        if not SHOWCASE.exists():
            raise SystemExit("Run assistant/evaluate.py first to create results/showcase.json")
        options = json.loads(SHOWCASE.read_text())[KINDS[args.showcase]]
        if not options:
            raise SystemExit(f"No examples of type '{args.showcase}' in this run. Try another type.")
        question = options[min(args.pick, len(options) - 1)]
    elif args.question:
        question = args.question
    else:
        raise SystemExit('Give a question in quotes, or use --showcase')

    lookup = {q["question"]: q for q in questions}
    gold = lookup.get(question)
    retriever = Retriever(passages, dates)

    console.print(f"\n[bold]Question:[/bold] {question}")
    if gold:
        console.print(f"[dim]Correct answer (from the dataset): {gold['answer']}  |  ★ = correct passage[/dim]\n")

    base, base_ret = answer_question(retriever, question, rerank=False, retry=True)
    show(f"1. BASELINE retrieval → {llm.model_label()} → checks", base, base_ret, gold and gold["gold_id"])

    ta, ta_ret = answer_question(retriever, question, rerank=True, retry=True)
    show(f"2. TIME-AWARE retrieval (our re-ranker) → {llm.model_label()} → checks", ta, ta_ret, gold and gold["gold_id"])


if __name__ == "__main__":
    main()

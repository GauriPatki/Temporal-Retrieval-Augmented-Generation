# Time-aware answer assistant (LLM + output checks)

This adds an answer step on top of the team's time-aware retriever. It answers the
"future work" item from our report: *does better time-aware retrieval actually give
more accurate final answers?*

```
question → retrieve (baseline or our time-aware re-ranker) → Claude answers in JSON
         → automatic checks → one retry with feedback → "passed" or "needs human review"
```

## What the checks catch

| Code | Meaning |
|---|---|
| `FORMAT` | Output isn't valid JSON with `answer`, `cited_ids`, `confidence` |
| `NO_CITATION` | Gave an answer without citing a passage |
| `INVENTED_SOURCE` | Cited a passage that was never retrieved |
| `WRONG_TIME` | Cited passage's years are far (>~10 years) from the question's year |
| `NOT_GROUNDED` | The answer's words don't appear in the cited passages |

## Results

See [`results/results.md`](results/results.md). It compares baseline retrieval + Claude,
time-aware retrieval + Claude, and time-aware + Claude + checks on year-explicit
ChroniclingAmericaQA questions.

## Run it

From the repo root:

```bash
pip install -r assistant/requirements.txt
export ANTHROPIC_API_KEY="your-key"

python -m pytest assistant/tests -q        # unit tests, no model or key needed
python assistant/evaluate.py --n 50        # builds the index once, runs the comparison
python assistant/demo.py --showcase        # side-by-side demo
```

Uses the public `facebook/contriever-msmarco` model and the team's re-ranker in
`src/mrag_integration.py`. Runs on a laptop CPU. Set `CLAUDE_MODEL` to change the Claude model.

**Free option:** install [Ollama](https://ollama.com), run `ollama pull qwen2.5:3b`, then set
`LLM_PROVIDER=ollama` (and optionally `OLLAMA_MODEL`). No API key needed.

## Files

- `retrieval.py`: loads the data, builds the index, baseline vs time-aware search
- `llm.py`: the prompt and the Claude call
- `checks.py`: the output checks
- `pipeline.py`: retrieve → answer → check → retry, logs every call to `runs/`
- `evaluate.py`: the three-way comparison
- `demo.py`: side-by-side terminal demo
- `tests/`: unit tests for the checks and scoring

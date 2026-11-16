"""
Checks that run on every Claude answer before anyone sees it.

Each failure is a string starting with a code:
  FORMAT          output is not valid JSON with the required fields
  EMPTY           answer is empty
  NO_CITATION     an answer was given without citing a passage
  INVENTED_SOURCE cites a passage id that was not retrieved
  WRONG_TIME      a cited passage's years are far from the question's year
  NOT_GROUNDED    the answer's words do not appear in the cited passages
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mrag_integration import compute_temporal_score, decompose_question_temporal_v1
from scoring import is_not_found, normalize_tokens

REQUIRED_FIELDS = {"answer", "cited_ids", "confidence"}
CONFIDENCE_VALUES = {"high", "medium", "low"}
MIN_TEMPORAL_SCORE = 0.5  # with the 20-year decay, this means within about 10 years
MIN_GROUNDED_SHARE = 0.6
STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or", "by",
    "with", "was", "is", "were", "be", "as", "from", "that", "this", "it", "his", "her",
}


def _tokens(text):
    return [t for t in normalize_tokens(text) if t not in STOPWORDS]


def parse_json(raw):
    """Pull the JSON object out of Claude's output. Returns (data, error)."""
    match = re.search(r"\{.*\}", raw or "", re.DOTALL)
    if not match:
        return None, "FORMAT: no JSON object in the output"
    try:
        return json.loads(match.group(0)), None
    except json.JSONDecodeError as e:
        return None, f"FORMAT: invalid JSON ({e.msg})"


def is_grounded(answer, cited_text):
    """True if most of the answer's content words appear in the cited text."""
    answer_tokens = _tokens(answer)
    if not answer_tokens:
        return True
    source = set(_tokens(cited_text))
    share = sum(t in source for t in answer_tokens) / len(answer_tokens)
    return share >= MIN_GROUNDED_SHARE


def _describe(tc):
    if tc.type == "range":
        return f"{tc.start_year}-{tc.end_year}"
    return str(tc.year)


def check_answer(question, raw, retrieved):
    """Returns (parsed_answer_or_None, list_of_failures). Empty list = passed."""
    data, error = parse_json(raw)
    if error:
        return None, [error]

    missing = REQUIRED_FIELDS - set(data)
    if missing:
        return data, [f"FORMAT: missing fields {sorted(missing)}"]

    failures = []
    if isinstance(data["answer"], (int, float)) and not isinstance(data["answer"], bool):
        data["answer"] = str(data["answer"])  # small models sometimes return 3 instead of "3"
    answer, cited = data["answer"], data["cited_ids"]

    if data["confidence"] not in CONFIDENCE_VALUES:
        failures.append("FORMAT: confidence must be high, medium or low")
    if not isinstance(cited, list) or not all(isinstance(c, int) for c in cited):
        return data, failures + ["FORMAT: cited_ids must be a list of integers"]
    if not isinstance(answer, str) or not answer.strip():
        return data, failures + ["EMPTY: the answer is empty"]
    if is_not_found(answer):
        data["answer"] = "NOT_FOUND"
        return data, failures

    by_id = {p["id"]: p for p in retrieved}
    if not cited:
        failures.append("NO_CITATION: the answer does not cite any passage")
    invented = [c for c in cited if c not in by_id]
    if invented:
        failures.append(f"INVENTED_SOURCE: cites passages that were not retrieved: {invented}")

    _, tc = decompose_question_temporal_v1(question)
    if tc is not None:
        for c in cited:
            if c in by_id and compute_temporal_score(tc, by_id[c]["years"]) < MIN_TEMPORAL_SCORE:
                failures.append(
                    f"WRONG_TIME: passage {c} mentions years {by_id[c]['years'][:5]}, "
                    f"but the question asks about {_describe(tc)}"
                )

    real = [c for c in cited if c in by_id]
    if real and not is_grounded(answer, " ".join(by_id[c]["text"] for c in real)):
        failures.append("NOT_GROUNDED: the answer's words do not appear in the cited passages")

    return data, failures

"""
LLM answer step: answer only from retrieved passages, return JSON.

Two providers, chosen with the LLM_PROVIDER environment variable:
  claude (default)  Anthropic API, needs ANTHROPIC_API_KEY
  ollama            free local model through Ollama (https://ollama.com), no key
"""

import json
import os
import re
import urllib.request

PROVIDER = os.environ.get("LLM_PROVIDER", "claude").lower()
MODEL = os.environ.get("CLAUDE_MODEL", "claude-haiku-5-5")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
# Local models on a CPU are slow with long prompts, so they get shorter excerpts.
MAX_PASSAGE_CHARS = int(os.environ.get("MAX_PASSAGE_CHARS", 1500 if PROVIDER == "ollama" else 3000))

SYSTEM_PROMPT = """You answer questions using ONLY the numbered newspaper passages provided.

Rules:
- Use only facts stated in the passages. Do not use outside knowledge.
- Each passage is from a newspaper page. A date in the question usually means the newspaper's
  publication date, which may NOT appear in the passage text itself.
- If a passage contains the answer and its publication date matches the question, answer it.
- If the passages do not contain the answer, set "answer" to "NOT_FOUND" and "cited_ids" to [].
- Cite the id of every passage you used.
- Keep the answer short: a name, date, number or short phrase, copied from the passage where possible.

Return ONLY this JSON object, with no other text:
{"answer": "...", "cited_ids": [<passage id numbers>], "confidence": "high" | "medium" | "low"}"""

_client = None


def _get_client():
    global _client
    if _client is None:
        import anthropic  # imported lazily so tests run without an API key

        _client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    return _client


def model_label():
    """Short name of the model in use, for tables and the demo."""
    return f"local {OLLAMA_MODEL}" if PROVIDER == "ollama" else f"Claude ({MODEL})"


def excerpt(text, question, limit=None):
    """Keep the part of a long passage that best matches the question, up to `limit` characters."""
    limit = limit or MAX_PASSAGE_CHARS
    if len(text) <= limit:
        return text
    sentences = re.split(r"(?<=[.!?])\s+", text)
    q_words = set(re.findall(r"[a-z0-9]+", question.lower())) - {"the", "a", "of", "in", "on", "what", "who", "when", "how", "was", "is"}
    scores = [len(q_words & set(re.findall(r"[a-z0-9]+", s.lower()))) for s in sentences]
    best = max(range(len(sentences)), key=scores.__getitem__)
    lo = hi = best
    size = len(sentences[best])
    while True:  # grow around the best sentence while it fits
        grew = False
        if hi + 1 < len(sentences) and size + len(sentences[hi + 1]) + 1 <= limit:
            hi += 1
            size += len(sentences[hi]) + 1
            grew = True
        if lo > 0 and size + len(sentences[lo - 1]) + 1 <= limit:
            lo -= 1
            size += len(sentences[lo]) + 1
            grew = True
        if not grew:
            break
    return " ".join(sentences[lo:hi + 1])[:limit]


def build_prompt(question, passages, feedback=None):
    blocks = []
    for p in passages:
        published = p.get("published") or "unknown"
        blocks.append(f"[Passage {p['id']}] (published: {published})\n{excerpt(p['text'], question)}")
    prompt = "Passages:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"
    if feedback:
        prompt += (
            "\n\nYour previous answer failed these automatic checks:\n- "
            + "\n- ".join(feedback)
            + "\nFix the problems and return the JSON again."
        )
    return prompt


def _ask_ollama(prompt):
    body = json.dumps({
        "model": OLLAMA_MODEL,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0, "num_ctx": 4096},
    }).encode()
    request = urllib.request.Request(OLLAMA_URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=900) as response:
        return json.loads(response.read())["message"]["content"]


def ask_claude(question, passages, feedback=None):
    """Returns the model's raw text output (Claude by default, or a local Ollama model)."""
    if PROVIDER == "ollama":
        return _ask_ollama(build_prompt(question, passages, feedback))
    response = _get_client().messages.create(
        model=MODEL,
        max_tokens=300,
        temperature=0,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_prompt(question, passages, feedback)}],
    )
    return "".join(block.text for block in response.content if block.type == "text")

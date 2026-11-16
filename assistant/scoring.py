"""Text normalization and loose answer matching, shared by the checks and the evaluation."""

import re

NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10", "eleven": "11",
    "twelve": "12", "thirteen": "13", "fourteen": "14", "fifteen": "15", "sixteen": "16",
    "seventeen": "17", "eighteen": "18", "nineteen": "19", "twenty": "20",
}


def normalize_tokens(text):
    """Lowercase words and numbers: '21,300' -> '21300', 'ONE' -> '1', curly quotes removed."""
    text = re.sub(r"(?<=\d),(?=\d)", "", ("" if text is None else str(text)).lower())
    return [NUMBER_WORDS.get(t, t) for t in re.findall(r"[a-z0-9]+", text)]


def is_not_found(answer):
    """True if the model said it could not find the answer (also catches '... NOT_FOUND ...')."""
    return "not_found" in ("" if answer is None else str(answer)).lower() or " ".join(normalize_tokens(answer)) == "not found"


def is_correct(pred, gold):
    """Loose match: one contains the other, or token F1 >= 0.5."""
    if is_not_found(pred):
        return False
    pt, gt = normalize_tokens(pred), normalize_tokens(gold)
    if not pt or not gt:
        return False
    p, g = " ".join(pt), " ".join(gt)
    if f" {g} " in f" {p} " or f" {p} " in f" {g} ":
        return True
    common = sum(min(pt.count(t), gt.count(t)) for t in set(pt))
    if common == 0:
        return False
    precision, recall = common / len(pt), common / len(gt)
    return 2 * precision * recall / (precision + recall) >= 0.5

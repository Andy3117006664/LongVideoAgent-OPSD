from __future__ import annotations

import re

from .schema import LETTERS


OPTION_RE = re.compile(r"(?m)^\s*\(([A-D])\)\s*(.+?)\s*$")


def parse_question(text: str) -> tuple[str, dict[str, str]]:
    matches = list(OPTION_RE.finditer(str(text or "")))
    options = {match.group(1): match.group(2).strip() for match in matches}
    if set(options) != set(LETTERS):
        raise ValueError("question must contain exactly four A-D options")
    stem = text[: matches[0].start()].strip()
    if not stem:
        raise ValueError("question stem is empty")
    return stem, options


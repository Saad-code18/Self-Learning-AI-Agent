"""
utils.py - small helpers shared by several modules.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any


def parse_json_object(text: str) -> dict[str, Any]:
    """
    Turn the model's answer into a Python dict, even if the model was sloppy.

    Models sometimes wrap JSON in ```json fences or add a sentence before it.
    We try, in order:
        1. plain json.loads
        2. strip ``` code fences and try again
        3. cut out the first {...} block we can find
    If everything fails we return {} so the caller can treat it as "no data"
    instead of crashing the chat.
    """
    if not text:
        return {}

    candidate = text.strip()

    # 1) The happy path (JSON mode normally lands here).
    try:
        data = json.loads(candidate)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        pass

    # 2) Remove ```json ... ``` fences.
    fenced = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate, flags=re.IGNORECASE | re.MULTILINE).strip()
    try:
        data = json.loads(fenced)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        pass

    # 3) Take the outermost {...} block.
    start, end = candidate.find("{"), candidate.rfind("}")
    if start != -1 and end > start:
        try:
            data = json.loads(candidate[start : end + 1])
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            pass

    return {}


def clean_fact(text: Any) -> str | None:
    """
    Normalise a fact string coming from the model:
    - must be a non-empty string
    - trim spaces and surrounding quotes
    - make sure it ends with a period so memories look consistent
    Returns None when the value is unusable.
    """
    if not isinstance(text, str):
        return None
    fact = text.strip().strip('"').strip("'").strip()
    if len(fact) < 3:
        return None
    if fact[-1] not in ".!?":
        fact += "."
    return fact


class Stopwatch:
    """
    Measure how long a block of code takes.

        with Stopwatch() as sw:
            do_something()
        print(sw.seconds)
    """

    def __enter__(self) -> "Stopwatch":
        self._start = time.perf_counter()
        self.seconds = 0.0
        return self

    def __exit__(self, *exc) -> None:
        self.seconds = time.perf_counter() - self._start


def estimate_tokens(text: str) -> int:
    """
    Rough token estimate (about 4 characters per token for English).
    Used only for on-screen hints; real numbers come from the API's `usage` field.
    """
    return max(1, len(text) // 4)

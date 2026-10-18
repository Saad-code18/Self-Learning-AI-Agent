"""
baseline.py - two "control group" agents used ONLY by the benchmark.

Why do we need them?
    To measure the value of the memory pipeline we compare it against what
    people usually build:

    1. StatelessRAGBaseline
         * No long-term memory.  Every new session starts empty.
         * Inside a session it does simple retrieval (RAG) over the raw
           messages of THAT session only (top-k by word overlap).
         -> shows the ACCURACY problem: it cannot recall anything from
            earlier sessions.

    2. FullHistoryBaseline
         * Remembers everything by stuffing the complete dialogue of ALL
           sessions into the prompt (the naive way to get "memory").
         -> shows the COST problem: the prompt grows with every message, so
            token usage, cost and latency explode.

    Our MemoryAgent (agent.py) sits in the middle: cross-session recall like
    #2, but with a prompt as small as #1 because only distilled facts are stored.

Both baselines expose the same interface as MemoryAgent
(`start_session`, `chat`, `wait_for_learning`, `close`) so the benchmark can
treat all three exactly the same way.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from memory_agent import prompts
from memory_agent.agent import ChatResult, SessionState
from memory_agent.llm import LLMClient, UsageTracker
from memory_agent.settings import Settings
from memory_agent.utils import Stopwatch

_WORD_RE = re.compile(r"[a-z0-9']+")
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "am", "are", "was", "were", "be", "to", "of", "in", "on", "at", "for",
    "i", "me", "my", "you", "your", "it", "this", "that", "with", "do", "does", "did", "have", "has", "can", "could",
    "what", "which", "who", "how", "any", "some", "about", "please", "tell", "know", "remind", "now", "just",
}


def _keywords(text: str) -> set[str]:
    """Lower-case words of a text without very common filler words."""
    return {w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS and len(w) > 1}


def lexical_top_k(query: str, documents: list[str], k: int) -> list[str]:
    """
    Tiny retriever: rank documents by how many query words they share.
    Good enough to simulate 'RAG over the current session' without embeddings.
    """
    q = _keywords(query)
    if not q or not documents:
        return []
    scored = []
    for doc in documents:
        overlap = len(q & _keywords(doc))
        if overlap > 0:
            scored.append((overlap, doc))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [doc for _, doc in scored[:k]]


@dataclass
class _BaseBaseline:
    """Shared plumbing for both baselines."""

    settings: Settings
    llm: LLMClient
    tracker: UsageTracker = field(init=False)

    def __post_init__(self) -> None:
        self.tracker = self.llm.tracker

    def start_session(self, user_id: str | None = None) -> SessionState:
        return SessionState(user_id=user_id or self.settings.default_user_id)

    def wait_for_learning(self) -> None:  # baselines have nothing to wait for
        return None

    def close(self) -> None:
        return None

    def _answer(self, system_prompt: str, window: list[dict], user_message: str, recalled_texts: list[str]) -> ChatResult:
        with Stopwatch() as clock:
            response = self.llm.chat(
                [{"role": "system", "content": system_prompt}, *window, {"role": "user", "content": user_message}],
                purpose="answer",
            )
        # Baselines return the recalled context as plain MemoryHit-like objects for the reports.
        from memory_agent.memory_store import MemoryHit  # local import to avoid a circular import

        return ChatResult(
            reply=response.text,
            recalled=[MemoryHit(id=str(i), text=t) for i, t in enumerate(recalled_texts)],
            learn=None,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            answer_seconds=clock.seconds,
            learn_seconds=0.0,
        )


class StatelessRAGBaseline(_BaseBaseline):
    """No memory across sessions; RAG over the current session's raw messages."""

    def chat(self, session: SessionState, user_message: str) -> ChatResult:
        user_message = user_message.strip()
        # "Knowledge base" = the raw user messages of THIS session only.
        documents = [m["content"] for m in session.messages if m["role"] == "user"]
        recalled = lexical_top_k(user_message, documents, k=self.settings.recall_top_k)
        system_prompt = prompts.render_baseline_system_prompt([f"- {t}" for t in recalled])
        window = session.messages[-self.settings.recent_messages_window :]

        result = self._answer(system_prompt, window, user_message, recalled)
        session.messages.extend([{"role": "user", "content": user_message}, {"role": "assistant", "content": result.reply}])
        session.turns += 1
        return result


class FullHistoryBaseline(_BaseBaseline):
    """Remembers everything by keeping the whole dialogue of every session in the prompt."""

    def __post_init__(self) -> None:
        super().__post_init__()
        # user_id -> complete transcript across ALL sessions (this is the "store full dialogue" strategy)
        self._history: dict[str, list[dict]] = {}

    def chat(self, session: SessionState, user_message: str) -> ChatResult:
        user_message = user_message.strip()
        history = self._history.setdefault(session.user_id, [])
        # The prompt contains the FULL transcript of all previous sessions ...
        context_lines = [f"{m['role']}: {m['content']}" for m in history]
        system_prompt = prompts.render_baseline_system_prompt(context_lines)
        # ... plus the messages of the current session.
        window = list(session.messages)

        result = self._answer(system_prompt, window, user_message, context_lines)
        turn = [{"role": "user", "content": user_message}, {"role": "assistant", "content": result.reply}]
        session.messages.extend(turn)
        session.turns += 1
        return result

    def end_session(self, session: SessionState) -> None:
        """Move the finished session into the permanent transcript."""
        self._history.setdefault(session.user_id, []).extend(session.messages)

    def forget_user(self, user_id: str) -> None:
        self._history.pop(user_id, None)

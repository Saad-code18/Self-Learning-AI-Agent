"""
agent.py - the self-learning conversational agent.

One chat turn looks like this:

    user message
        │
        ▼
    1. RECALL   : search Mem0/Qdrant for memories related to the message
    2. ANSWER   : system prompt (persona + recalled memories)
                  + last few messages of THIS session + the user message  -> LLM
    3. LEARN    : run the two-phase memory pipeline on [user message, reply]
                  (synchronously, or in a background thread so the user is not kept waiting)

Sessions vs memories:
    * A `SessionState` is short-lived: it holds the messages of the current
      chat window and a rolling summary.  It is NOT persisted.
    * Memories live in Qdrant and survive restarts -> that is what makes the
      agent remember you across sessions ("cross-session recall").
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field

from memory_agent import prompts
from memory_agent.llm import LLMClient, UsageTracker
from memory_agent.memory_store import MemoryHit, MemoryStore
from memory_agent.pipeline import LearnResult, MemoryEvent, MemoryPipeline
from memory_agent.settings import Settings, get_settings
from memory_agent.utils import Stopwatch

logger = logging.getLogger(__name__)


@dataclass
class SessionState:
    """Everything the agent knows about the CURRENT chat window (not persisted)."""

    user_id: str
    messages: list[dict] = field(default_factory=list)  # full transcript of this session
    summary: str = ""  # rolling summary produced by the pipeline
    turns: int = 0  # number of user messages so far
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


@dataclass
class ChatResult:
    """What `MemoryAgent.chat()` returns - the reply plus everything useful for logs/UI/benchmark."""

    reply: str
    recalled: list[MemoryHit]
    learn: LearnResult | None  # None when learning runs in the background
    prompt_tokens: int  # tokens of the ANSWER call only
    completion_tokens: int
    answer_seconds: float  # time the user waited for the reply
    learn_seconds: float  # time spent learning (0 when in background)


class MemoryAgent:
    """The memory-powered assistant."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        llm: LLMClient | None = None,
        store: MemoryStore | None = None,
        pipeline: MemoryPipeline | None = None,
        tracker: UsageTracker | None = None,
    ):
        self.settings = settings or get_settings()
        self.tracker = tracker or UsageTracker(self.settings.price_input_per_1m, self.settings.price_output_per_1m)
        self.llm = llm or LLMClient(self.settings, tracker=self.tracker)
        self.store = store or MemoryStore(self.settings)
        self.pipeline = pipeline or MemoryPipeline(self.llm, self.store, self.settings)

        # One worker thread is enough: learning steps for the same user must not
        # race each other (two parallel UPDATEs could otherwise fight).
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="memory-learn")
        self._pending: list[Future] = []

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------
    def start_session(self, user_id: str | None = None) -> SessionState:
        """Begin a new chat window for a user. Memories from older sessions stay available."""
        return SessionState(user_id=user_id or self.settings.default_user_id)

    # ------------------------------------------------------------------
    # Step 1 - RECALL
    # ------------------------------------------------------------------
    def recall(self, user_id: str, query: str) -> list[MemoryHit]:
        """Fetch the memories most related to the user's message."""
        try:
            return self.store.search(query, user_id=user_id, top_k=self.settings.recall_top_k)
        except Exception as exc:  # noqa: BLE001 - a memory outage must not kill the chat
            logger.error("Memory search failed, answering without memories: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Step 2 - ANSWER
    # ------------------------------------------------------------------
    def _recent_window(self, session: SessionState) -> list[dict]:
        """The last N messages of this session (N = RECENT_MESSAGES_WINDOW; 0 means none)."""
        n = self.settings.recent_messages_window
        return list(session.messages[-n:]) if n > 0 else []

    def build_messages(self, session: SessionState, recalled: list[MemoryHit], user_message: str) -> list[dict]:
        """Assemble the prompt: persona + memories, a short window of this session, the new message."""
        system_prompt = prompts.render_answer_system_prompt([m.text for m in recalled])
        return [{"role": "system", "content": system_prompt}, *self._recent_window(session), {"role": "user", "content": user_message}]

    def chat(self, session: SessionState, user_message: str) -> ChatResult:
        """Run one full turn: recall -> answer -> learn."""
        user_message = user_message.strip()
        recalled = self.recall(session.user_id, user_message)

        with Stopwatch() as answer_clock:
            response = self.llm.chat(self.build_messages(session, recalled, user_message), purpose="answer")
        reply = response.text

        # Remember this turn inside the session window.
        turn_messages = [{"role": "user", "content": user_message}, {"role": "assistant", "content": reply}]
        with session.lock:
            # The messages BEFORE this turn are the context for reference resolution.
            recent_for_learning = self._recent_window(session)
            session.messages.extend(turn_messages)
            session.turns += 1
            refresh_summary = session.turns % self.settings.summary_refresh_every == 0

        # Step 3 - LEARN (sync or background)
        learn_result: LearnResult | None = None
        learn_seconds = 0.0
        if self.settings.learn_in_background:
            self._pending.append(
                self._executor.submit(self._learn_safely, session, turn_messages, recent_for_learning, refresh_summary)
            )
            self._pending = [f for f in self._pending if not f.done()]  # forget finished jobs
        else:
            with Stopwatch() as learn_clock:
                learn_result = self._learn_safely(session, turn_messages, recent_for_learning, refresh_summary)
            learn_seconds = learn_clock.seconds

        return ChatResult(
            reply=reply,
            recalled=recalled,
            learn=learn_result,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            answer_seconds=answer_clock.seconds,
            learn_seconds=learn_seconds,
        )

    # ------------------------------------------------------------------
    # Step 3 - LEARN
    # ------------------------------------------------------------------
    def learn(self, session: SessionState, turn_messages: list[dict], recent_messages: list[dict], refresh_summary: bool) -> LearnResult:
        """Run the memory pipeline for one turn and update the session summary."""
        if self.settings.memory_pipeline == "mem0":
            # Mem0's built-in extraction (the tutorial-video approach). One call does everything.
            raw_events = self.store.learn_with_mem0(turn_messages, session.user_id)
            events = [MemoryEvent(str(e.get("event", "ADD")).upper(), str(e.get("memory", "")), e.get("id")) for e in raw_events]
            return LearnResult(facts=[e.text for e in events], events=events, summary=session.summary)

        result = self.pipeline.learn(
            turn_messages,
            session.user_id,
            summary=session.summary,
            recent_messages=recent_messages,
            refresh_summary=refresh_summary,
        )
        if result.summary_refreshed:
            with session.lock:
                session.summary = result.summary
        return result

    def _learn_safely(self, session: SessionState, turn_messages: list[dict], recent: list[dict], refresh_summary: bool) -> LearnResult:
        """Learning must never crash the chat: log the error and carry on."""
        try:
            return self.learn(session, turn_messages, recent, refresh_summary)
        except Exception as exc:  # noqa: BLE001
            logger.error("Learning step failed: %s", exc)
            return LearnResult(summary=session.summary)

    def wait_for_learning(self) -> None:
        """Block until all background learning jobs finished (used on exit and in the benchmark)."""
        for future in list(self._pending):
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001
                logger.error("Background learning job failed: %s", exc)
        self._pending.clear()

    # ------------------------------------------------------------------
    # Convenience helpers used by the CLI / API
    # ------------------------------------------------------------------
    def list_memories(self, user_id: str) -> list[MemoryHit]:
        return self.store.list_all(user_id)

    def forget_user(self, user_id: str) -> None:
        self.store.delete_all(user_id)

    def close(self) -> None:
        """Finish background work and stop the worker thread."""
        self.wait_for_learning()
        self._executor.shutdown(wait=True)

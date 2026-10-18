"""
conftest.py - shared test helpers (pytest loads this file automatically).

The unit tests must run WITHOUT an API key and WITHOUT Qdrant, so we replace
the two external pieces with fakes:

    FakeLLM         - returns scripted answers (per "purpose") instead of calling a model
    FakeMemoryStore - keeps memories in a Python dict and searches by word overlap

Both expose the same methods as the real classes, so the pipeline and the
agent run exactly the same code paths as in production.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field

import pytest

from memory_agent.llm import LLMResponse, UsageTracker
from memory_agent.memory_store import MemoryHit
from memory_agent.settings import Settings


def make_settings(**overrides) -> Settings:
    """A Settings object with test-friendly defaults (no .env needed)."""
    values = dict(
        llm_provider="openai",
        llm_model="fake-model",
        openai_api_key="sk-test",
        openai_base_url=None,
        ollama_base_url="http://localhost:11434",
        embedder_provider="openai",
        embed_model="fake-embed",
        embed_dims=64,
        qdrant_host="localhost",
        qdrant_port=6333,
        qdrant_collection="test_collection",
        history_db_path="/tmp/test_history.db",
        memory_pipeline="custom",
        recall_top_k=5,
        similar_top_k=5,
        summary_refresh_every=2,
        recent_messages_window=4,
        learn_in_background=False,
        price_input_per_1m=1.0,
        price_output_per_1m=2.0,
        log_level="ERROR",
        default_user_id="tester",
    )
    values.update(overrides)
    return Settings(**values)


@dataclass
class FakeLLM:
    """
    Scripted stand-in for LLMClient.

    `responses` maps a purpose ("answer", "extract", "decide", "summarize") to a
    list of answers returned in order (the last one repeats forever).
    Every call is recorded in `calls` so tests can inspect the prompts.
    """

    responses: dict = field(default_factory=dict)
    tracker: UsageTracker = field(default_factory=lambda: UsageTracker(1.0, 2.0))
    calls: list = field(default_factory=list)
    _cursor: dict = field(default_factory=dict)

    def chat(self, messages, *, json_mode=False, temperature=0.2, max_tokens=800, purpose="answer"):
        self.calls.append({"purpose": purpose, "messages": messages, "json_mode": json_mode})
        options = self.responses.get(purpose, ["fake answer"])
        idx = min(self._cursor.get(purpose, 0), len(options) - 1)
        self._cursor[purpose] = idx + 1
        text = options[idx]
        if not isinstance(text, str):
            text = json.dumps(text)  # allow dicts for JSON answers
        prompt_tokens = sum(len(m["content"]) for m in messages) // 4
        response = LLMResponse(text=text, prompt_tokens=prompt_tokens, completion_tokens=len(text) // 4, latency_seconds=0.001)
        self.tracker.record(purpose, response)
        return response


class FakeMemoryStore:
    """In-memory replacement for MemoryStore (dict + word-overlap search)."""

    def __init__(self):
        self.memories: dict[str, dict] = {}  # id -> {"text", "user_id"}
        self.log: list[tuple] = []  # every write, for assertions

    # --- writes ---
    def add_fact(self, text, user_id, metadata=None):
        mem_id = str(uuid.uuid4())
        self.memories[mem_id] = {"text": text, "user_id": user_id}
        self.log.append(("ADD", mem_id, text))
        return mem_id

    def add_facts(self, texts, user_id, metadata=None):
        return [self.add_fact(t, user_id, metadata) for t in texts]

    def update_fact(self, memory_id, new_text):
        self.memories[memory_id]["text"] = new_text
        self.log.append(("UPDATE", memory_id, new_text))

    def delete_fact(self, memory_id):
        self.memories.pop(memory_id, None)
        self.log.append(("DELETE", memory_id, None))

    def delete_all(self, user_id):
        for mid in [m for m, v in self.memories.items() if v["user_id"] == user_id]:
            del self.memories[mid]

    # --- reads ---
    @staticmethod
    def _words(text):
        return set(re.findall(r"[a-z0-9']+", text.lower()))

    def search(self, query, user_id, top_k=5):
        words = self._words(query)
        scored = []
        for mid, m in self.memories.items():
            if m["user_id"] != user_id:
                continue
            overlap = len(words & self._words(m["text"]))
            if overlap:
                scored.append((overlap, mid))
        scored.sort(reverse=True)
        return [MemoryHit(id=mid, text=self.memories[mid]["text"], score=float(s)) for s, mid in scored[:top_k]]

    def list_all(self, user_id, limit=10_000):
        return [MemoryHit(id=mid, text=m["text"]) for mid, m in self.memories.items() if m["user_id"] == user_id]

    def count(self, user_id=None):
        return sum(1 for m in self.memories.values() if user_id is None or m["user_id"] == user_id)

    def ping(self):
        return True

    def learn_with_mem0(self, messages, user_id, custom_instructions=None):
        mid = self.add_fact("mem0 native fact.", user_id)
        return [{"id": mid, "memory": "mem0 native fact.", "event": "ADD"}]

    def texts(self, user_id):
        return sorted(m["text"] for m in self.memories.values() if m["user_id"] == user_id)


@pytest.fixture
def settings():
    return make_settings()


@pytest.fixture
def store():
    return FakeMemoryStore()

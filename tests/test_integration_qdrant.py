"""
Integration test against a REAL Qdrant server (Mem0 + Qdrant wiring).

It is skipped automatically unless Qdrant is reachable AND you opt in:

    RUN_INTEGRATION=1 pytest tests/test_integration_qdrant.py -v

No API key is needed: the embedder is swapped for a deterministic fake so the
test only checks the storage/search/update/delete/history plumbing.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import uuid

import pytest

from conftest import make_settings

pytestmark = pytest.mark.integration


class HashEmbedder:
    """Deterministic 'embedding': bag of hashed words. Same text -> same vector."""

    def __init__(self, dims: int = 64):
        self.dims = dims

    @staticmethod
    def _tokens(text):
        # crude normalisation so "movies." and "loves" match "movie" / "love"
        return [w.rstrip("s") for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 1]

    def embed(self, text, memory_action=None):
        vector = [0.0] * self.dims
        for word in self._tokens(text):
            vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dims] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed_batch(self, texts, memory_action=None):
        return [self.embed(t) for t in texts]


def _qdrant_up(host: str, port: int) -> bool:
    import socket

    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


@pytest.fixture
def real_store(tmp_path):
    settings = make_settings(
        qdrant_host=os.getenv("QDRANT_HOST", "localhost"),
        qdrant_port=int(os.getenv("QDRANT_PORT", "6333")),
        qdrant_collection=f"pytest_{uuid.uuid4().hex[:8]}",
        history_db_path=str(tmp_path / "history.db"),
        embed_dims=64,
    )
    if os.getenv("RUN_INTEGRATION") != "1":
        pytest.skip("set RUN_INTEGRATION=1 to run against a real Qdrant")
    if not _qdrant_up(settings.qdrant_host, settings.qdrant_port):
        pytest.skip("Qdrant is not running")

    from memory_agent.memory_store import MemoryStore

    store = MemoryStore(settings)
    store.memory.embedding_model = HashEmbedder(64)  # no API key needed
    yield store
    store.memory.vector_store.client.delete_collection(settings.qdrant_collection)


def test_add_search_update_delete_history_roundtrip(real_store):
    store = real_store
    assert store.ping()

    mid = store.add_fact("User loves sci-fi movies.", user_id="alice", metadata={"source": "test"})
    store.add_fact("User lives in Berlin.", user_id="alice")
    store.add_fact("User is a nurse.", user_id="bob")
    assert mid is not None
    assert store.count("alice") == 2 and store.count("bob") == 1 and store.count() == 3

    hits = store.search("which movies does the user love", user_id="alice", top_k=5)
    assert hits and hits[0].text == "User loves sci-fi movies."
    assert all(h.score is not None for h in hits)

    store.update_fact(mid, "User loves sci-fi and fantasy movies.")
    assert any(h.text == "User loves sci-fi and fantasy movies." for h in store.list_all("alice"))
    events = [h["event"] for h in store.history(mid)]
    assert events == ["ADD", "UPDATE"]

    store.delete_fact(mid)
    assert store.count("alice") == 1

    ids = store.add_facts(["User likes tea.", "User likes coffee."], user_id="alice")
    assert len(ids) == 2 and store.count("alice") == 3

    store.delete_all("alice")
    assert store.count("alice") == 0 and store.count("bob") == 1


def test_pipeline_end_to_end_with_real_store(real_store):
    """The whole agent loop with a scripted LLM but REAL Mem0/Qdrant storage."""
    from conftest import FakeLLM

    from memory_agent.agent import MemoryAgent
    from memory_agent.pipeline import MemoryPipeline

    llm = FakeLLM(
        responses={
            "answer": ["Hi Priya!", "Enjoy Munich!", "You live in Munich."],
            "extract": [
                {"facts": ["User's name is Priya.", "User lives in Berlin."]},
                {"facts": ["User lives in Munich."]},
                {"facts": []},
            ],
            "decide": [{"actions": [{"event": "UPDATE", "id": "0", "text": "User lives in Munich."}]}],
        }
    )
    settings = real_store.settings
    agent = MemoryAgent(settings, llm=llm, store=real_store, pipeline=MemoryPipeline(llm, real_store, settings), tracker=llm.tracker)

    session = agent.start_session("priya")
    agent.chat(session, "Hi, I'm Priya and I live in Berlin")
    assert sorted(m.text for m in real_store.list_all("priya")) == ["User lives in Berlin.", "User's name is Priya."]

    r2 = agent.chat(session, "I moved to Munich")
    assert [e.event for e in r2.learn.events] == ["UPDATE"]
    texts = sorted(m.text for m in real_store.list_all("priya"))
    assert texts == ["User lives in Munich.", "User's name is Priya."]

    # A NEW session must still recall the memories (cross-session recall).
    session2 = agent.start_session("priya")
    r3 = agent.chat(session2, "Where do I live?")
    assert any("Munich" in m.text for m in r3.recalled)
    agent.close()

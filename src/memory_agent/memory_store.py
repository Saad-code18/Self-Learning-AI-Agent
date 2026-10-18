"""
memory_store.py - the persistent long-term memory layer (Mem0 + Qdrant).

What lives here:
    * `build_mem0_config()`  - translates our Settings into the dict Mem0 expects
    * `MemoryStore`          - a small, friendly API on top of Mem0:
                                 add_fact / update_fact / delete_fact / search /
                                 list_all / count / delete_all / history

Why wrap Mem0 instead of calling it directly everywhere?
    * Mem0's API changed a lot between versions (for example `search()` now
      takes `filters={"user_id": ...}` instead of `user_id=...`).  Keeping all
      Mem0 calls in ONE file means only this file needs to change if Mem0
      changes again.
    * The rest of the code gets simple Python objects (`MemoryHit`) instead of
      raw dicts.

Storage layout (all self-hosted, nothing leaves your machine except LLM calls):
    * Qdrant  (Docker)  -> vectors + payload of every memory  (collection = QDRANT_COLLECTION)
    * SQLite  (file)    -> history of every ADD/UPDATE/DELETE   (HISTORY_DB_PATH)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from memory_agent.settings import Settings  # noqa: F401  (must be imported before mem0: disables telemetry)
from mem0 import Memory
from qdrant_client import models as qmodels

logger = logging.getLogger(__name__)


@dataclass
class MemoryHit:
    """One memory returned by a search or a listing."""

    id: str  # Mem0's UUID for this memory
    text: str  # the distilled fact, e.g. "User is vegetarian."
    score: float | None = None  # similarity score (only for searches)
    created_at: str | None = None
    updated_at: str | None = None
    metadata: dict | None = None


def build_mem0_config(settings: Settings) -> dict[str, Any]:
    """
    Build the configuration dict for `Memory.from_config()`.

    Three building blocks:
      vector_store -> Qdrant (where memories are stored and searched)
      embedder     -> turns text into vectors (OpenAI or Ollama)
      llm          -> only used when MEMORY_PIPELINE=mem0 (Mem0's own extraction);
                      our custom pipeline uses its own LLMClient instead.
    """
    if settings.embedder_provider == "ollama":
        embedder = {
            "provider": "ollama",
            "config": {
                "model": settings.embed_model,
                "embedding_dims": settings.embed_dims,
                "ollama_base_url": settings.ollama_base_url,
            },
        }
    else:
        embedder = {
            "provider": "openai",
            "config": {
                "model": settings.embed_model,
                "embedding_dims": settings.embed_dims,
                "api_key": settings.openai_api_key,
                "openai_base_url": settings.openai_base_url,
            },
        }

    if settings.llm_provider == "ollama":
        llm = {
            "provider": "ollama",
            "config": {
                "model": settings.llm_model,
                "ollama_base_url": settings.ollama_base_url,
                "temperature": 0.1,
            },
        }
    else:
        llm = {
            "provider": "openai",
            "config": {
                "model": settings.llm_model,
                "api_key": settings.openai_api_key,
                "openai_base_url": settings.openai_base_url,
                "temperature": 0.1,
            },
        }

    return {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "host": settings.qdrant_host,
                "port": settings.qdrant_port,
                "collection_name": settings.qdrant_collection,
                # Qdrant needs to know the vector size when it creates the collection.
                # It MUST match the embedding model (1536 for text-embedding-3-small).
                "embedding_model_dims": settings.embed_dims,
            },
        },
        "embedder": embedder,
        "llm": llm,
        "history_db_path": settings.history_db_path,
    }


class MemoryStore:
    """Friendly wrapper around Mem0's `Memory` object."""

    def __init__(self, settings: Settings, memory: Memory | None = None):
        self.settings = settings
        # `memory` can be injected for tests; normally we build it from settings.
        self.memory = memory or Memory.from_config(build_mem0_config(settings))
        self.collection_name = settings.qdrant_collection

    # ------------------------------------------------------------------
    # Writing memories (used by the pipeline's ADD / UPDATE / DELETE step)
    # ------------------------------------------------------------------
    def add_fact(self, text: str, user_id: str, metadata: dict | None = None) -> str | None:
        """
        Store ONE distilled fact exactly as given.

        `infer=False` tells Mem0: "do not run your own LLM extraction, just
        embed and save this text".  Our custom pipeline already did the
        thinking, so this only costs one embedding call (no LLM tokens).
        Returns the new memory id (or None if Mem0 stored nothing).
        """
        result = self.memory.add(text, user_id=user_id, metadata=metadata or {}, infer=False)
        results = result.get("results", []) if isinstance(result, dict) else []
        return results[0]["id"] if results else None

    def add_facts(self, texts: list[str], user_id: str, metadata: dict | None = None) -> list[str]:
        """Store several facts in one Mem0 call (used by the seed script)."""
        if not texts:
            return []
        messages = [{"role": "user", "content": t} for t in texts]
        result = self.memory.add(messages, user_id=user_id, metadata=metadata or {}, infer=False)
        return [r["id"] for r in result.get("results", [])]

    def update_fact(self, memory_id: str, new_text: str) -> None:
        """Replace the text of an existing memory (Mem0 re-embeds it and logs history)."""
        self.memory.update(memory_id, new_text)

    def delete_fact(self, memory_id: str) -> None:
        """Remove one memory (Mem0 keeps a 'DELETE' entry in the history table)."""
        self.memory.delete(memory_id)

    def delete_all(self, user_id: str) -> None:
        """Forget everything about one user."""
        self.memory.delete_all(user_id=user_id)

    # ------------------------------------------------------------------
    # Reading memories
    # ------------------------------------------------------------------
    def search(self, query: str, user_id: str, top_k: int | None = None) -> list[MemoryHit]:
        """
        Semantic search: 'what do I know about this user that relates to the query?'
        Mem0 embeds the query, asks Qdrant for the nearest vectors and returns them.
        """
        top_k = top_k or self.settings.recall_top_k
        result = self.memory.search(query, filters={"user_id": user_id}, top_k=top_k)
        return [self._to_hit(r) for r in result.get("results", [])]

    def list_all(self, user_id: str, limit: int = 10_000) -> list[MemoryHit]:
        """Every memory of one user (newest first is NOT guaranteed, we sort by created_at)."""
        result = self.memory.get_all(filters={"user_id": user_id}, top_k=limit)
        hits = [self._to_hit(r) for r in result.get("results", [])]
        hits.sort(key=lambda h: h.created_at or "")
        return hits

    def count(self, user_id: str | None = None) -> int:
        """
        Exact number of stored memories, for one user or for everybody.
        We ask Qdrant directly because Mem0 has no count() function.
        """
        client = self.memory.vector_store.client
        count_filter = None
        if user_id:
            count_filter = qmodels.Filter(
                must=[qmodels.FieldCondition(key="user_id", match=qmodels.MatchValue(value=user_id))]
            )
        return client.count(collection_name=self.collection_name, count_filter=count_filter, exact=True).count

    def history(self, memory_id: str) -> list[dict]:
        """Audit trail of one memory: every ADD / UPDATE / DELETE with old and new text."""
        return self.memory.history(memory_id)

    # ------------------------------------------------------------------
    # Mem0's built-in pipeline (MEMORY_PIPELINE=mem0)
    # ------------------------------------------------------------------
    def learn_with_mem0(self, messages: list[dict], user_id: str, custom_instructions: str | None = None) -> list[dict]:
        """
        Let Mem0 extract memories itself (`infer=True`).  This is the approach
        shown in the tutorial video.  Returns Mem0's list of events.
        """
        result = self.memory.add(messages, user_id=user_id, infer=True, prompt=custom_instructions)
        return result.get("results", [])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _to_hit(raw: dict) -> MemoryHit:
        """Convert Mem0's raw dict into our MemoryHit dataclass."""
        return MemoryHit(
            id=str(raw.get("id")),
            text=str(raw.get("memory", "")),
            score=raw.get("score"),
            created_at=raw.get("created_at"),
            updated_at=raw.get("updated_at"),
            metadata=raw.get("metadata"),
        )

    def ping(self) -> bool:
        """True when Qdrant answers. Used by the setup-check script and /health."""
        try:
            self.memory.vector_store.client.get_collections()
            return True
        except Exception as exc:  # noqa: BLE001 - we really want to catch everything here
            logger.error("Cannot reach Qdrant at %s:%s (%s)", self.settings.qdrant_host, self.settings.qdrant_port, exc)
            return False

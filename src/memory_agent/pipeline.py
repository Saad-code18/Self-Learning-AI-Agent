"""
pipeline.py - the TWO-PHASE MEMORY PIPELINE (the heart of the project).

After every conversation turn the agent calls `MemoryPipeline.learn(...)`:

    PHASE 1 - EXTRACTION
        1a. (every N turns) refresh a short rolling SUMMARY of the conversation
        1b. run the custom FACT-EXTRACTION prompt on the newest messages
            -> a list of short, standalone facts ("User is vegetarian.")

    PHASE 2 - UPDATE / DECISIONING
        2a. for each fact, fetch the most similar memories already stored
        2b. run the custom UPDATE prompt once: for every fact the model decides
                ADD     -> store as a new memory
                UPDATE  -> rewrite an existing memory (keeps its id + history)
                DELETE  -> remove a contradicted memory
                NONE    -> already known, do nothing
        2c. apply those decisions to Mem0/Qdrant

Why do this instead of saving the whole chat?
    * Only distilled facts are stored, so the answer prompt later contains a
      handful of short sentences instead of pages of dialogue -> far fewer
      prompt tokens, lower cost and lower latency.
    * The UPDATE/DELETE step keeps the store consistent when the user's life
      changes ("I moved to Denver") instead of piling up contradictions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from memory_agent import prompts
from memory_agent.llm import LLMClient
from memory_agent.memory_store import MemoryHit, MemoryStore
from memory_agent.settings import Settings
from memory_agent.utils import clean_fact, parse_json_object

logger = logging.getLogger(__name__)

VALID_EVENTS = ("ADD", "UPDATE", "DELETE", "NONE")


@dataclass
class MemoryEvent:
    """One change (or non-change) made to the memory store. Used for logs and the UI."""

    event: str  # ADD / UPDATE / DELETE / NONE
    text: str  # the fact text (new text for ADD/UPDATE, old text for DELETE/NONE)
    memory_id: str | None = None
    old_text: str | None = None  # only for UPDATE

    def describe(self) -> str:
        """Human friendly one-liner, e.g. UPDATE: 'User lives in Berlin' -> 'User lives in Munich'."""
        if self.event == "UPDATE" and self.old_text:
            return f"UPDATE: {self.old_text!r} -> {self.text!r}"
        return f"{self.event}: {self.text!r}"


@dataclass
class LearnResult:
    """Everything that happened during one `learn()` call (handy for the benchmark)."""

    facts: list[str] = field(default_factory=list)
    events: list[MemoryEvent] = field(default_factory=list)
    summary: str = ""
    summary_refreshed: bool = False

    @property
    def changed(self) -> int:
        """How many memories were actually added/updated/deleted."""
        return sum(1 for e in self.events if e.event != "NONE")


class MemoryPipeline:
    """Runs the two phases described at the top of this file."""

    def __init__(self, llm: LLMClient, store: MemoryStore, settings: Settings):
        self.llm = llm
        self.store = store
        self.settings = settings

    # ==================================================================
    # PHASE 1a - rolling summary
    # ==================================================================
    def summarize(self, previous_summary: str, new_messages: list[dict]) -> str:
        """Ask the model for an updated 120-word summary of the conversation."""
        response = self.llm.chat(
            [
                {"role": "system", "content": prompts.SUMMARY_PROMPT},
                {"role": "user", "content": prompts.render_summary_request(previous_summary, new_messages)},
            ],
            temperature=0.1,
            max_tokens=250,
            purpose="summarize",
        )
        return response.text.strip() or previous_summary

    # ==================================================================
    # PHASE 1b - fact extraction
    # ==================================================================
    def extract_facts(self, new_messages: list[dict], summary: str = "", recent_messages: list[dict] | None = None) -> list[str]:
        """
        Turn the newest messages into a list of standalone facts.
        The summary and recent messages are ONLY context (to resolve "it", "there"...).
        """
        response = self.llm.chat(
            [
                {"role": "system", "content": prompts.FACT_EXTRACTION_PROMPT},
                {
                    "role": "user",
                    "content": prompts.render_fact_extraction_request(summary, recent_messages or [], new_messages),
                },
            ],
            json_mode=True,
            temperature=0.0,
            max_tokens=600,
            purpose="extract",
        )
        data = parse_json_object(response.text)
        raw_facts = data.get("facts", [])
        if not isinstance(raw_facts, list):
            logger.warning("Fact extraction returned no list: %r", response.text[:200])
            return []

        facts: list[str] = []
        for raw in raw_facts:
            fact = clean_fact(raw)
            # Skip empty values and exact duplicates inside the same batch.
            if fact and fact.lower() not in {f.lower() for f in facts}:
                facts.append(fact)
        logger.info("Extracted %d fact(s): %s", len(facts), facts)
        return facts

    # ==================================================================
    # PHASE 2 - decide ADD / UPDATE / DELETE / NONE and apply
    # ==================================================================
    def decide_and_apply(self, facts: list[str], user_id: str) -> list[MemoryEvent]:
        """
        Compare the new facts with what is already stored and apply the result.
        Returns the list of MemoryEvents (including NONE for 'already known').
        """
        if not facts:
            return []

        # ---- 2a. Gather the memories that are similar to ANY of the new facts ----
        # We look up each fact separately, then merge the results (no duplicates).
        candidates: dict[str, MemoryHit] = {}
        for fact in facts:
            for hit in self.store.search(fact, user_id=user_id, top_k=self.settings.similar_top_k):
                candidates[hit.id] = hit

        # ---- Shortcut: nothing stored yet -> everything is an ADD (saves one LLM call) ----
        if not candidates:
            return [self._apply(MemoryEvent("ADD", fact), user_id) for fact in facts]

        # ---- 2b. Ask the model to decide.  Real UUIDs are replaced by "0", "1", "2"...
        #          because models copy short ids reliably and long UUIDs badly. ----
        index_to_id: dict[str, str] = {}
        existing_for_prompt: list[dict] = []
        for idx, hit in enumerate(candidates.values()):
            index_to_id[str(idx)] = hit.id
            existing_for_prompt.append({"id": str(idx), "text": hit.text})

        response = self.llm.chat(
            [
                {"role": "system", "content": prompts.UPDATE_DECISION_PROMPT},
                {"role": "user", "content": prompts.render_update_decision_request(existing_for_prompt, facts)},
            ],
            json_mode=True,
            temperature=0.0,
            max_tokens=800,
            purpose="decide",
        )
        actions = parse_json_object(response.text).get("actions", [])
        if not isinstance(actions, list) or not actions:
            # The model gave us nothing usable. Be safe and just ADD the facts:
            # a duplicate memory is better than a lost one.
            logger.warning("Decision step returned no actions, falling back to ADD for all facts.")
            return [self._apply(MemoryEvent("ADD", fact), user_id) for fact in facts]

        # ---- 2c. Validate every action and apply it ----
        events: list[MemoryEvent] = []
        for action in actions:
            event = self._validate_action(action, index_to_id, candidates)
            if event is None:
                continue
            events.append(self._apply(event, user_id))
        return events

    # ------------------------------------------------------------------
    # PHASE 2 helpers
    # ------------------------------------------------------------------
    def _validate_action(self, action: dict, index_to_id: dict[str, str], candidates: dict[str, MemoryHit]) -> MemoryEvent | None:
        """
        Make sure the model's action is safe to execute.
        Bad or unknown ids are handled gracefully:
          * UPDATE with unknown id  -> becomes ADD (we still keep the information)
          * DELETE/NONE with unknown id -> ignored
        """
        if not isinstance(action, dict):
            return None
        event = str(action.get("event", "")).upper().strip()
        if event not in VALID_EVENTS:
            logger.warning("Ignoring unknown event %r", event)
            return None

        idx = str(action.get("id", "")).strip()
        memory_id = index_to_id.get(idx)
        text = clean_fact(action.get("text"))

        if event == "ADD":
            return MemoryEvent("ADD", text) if text else None

        if event == "UPDATE":
            if not text:
                return None
            if memory_id is None:
                logger.warning("UPDATE with unknown id %r -> treating as ADD", idx)
                return MemoryEvent("ADD", text)
            old_text = candidates[memory_id].text
            if old_text.strip().lower() == text.strip().lower():
                return MemoryEvent("NONE", old_text, memory_id)  # nothing really changed
            return MemoryEvent("UPDATE", text, memory_id, old_text=old_text)

        if event == "DELETE":
            if memory_id is None:
                logger.warning("DELETE with unknown id %r -> ignored", idx)
                return None
            return MemoryEvent("DELETE", candidates[memory_id].text, memory_id)

        # NONE: already known
        if memory_id is None:
            return None
        return MemoryEvent("NONE", candidates[memory_id].text, memory_id)

    def _apply(self, event: MemoryEvent, user_id: str) -> MemoryEvent:
        """Execute one decision against Mem0/Qdrant and return the (possibly enriched) event."""
        if event.event == "ADD":
            event.memory_id = self.store.add_fact(event.text, user_id=user_id, metadata={"source": "pipeline"})
        elif event.event == "UPDATE" and event.memory_id:
            self.store.update_fact(event.memory_id, event.text)
        elif event.event == "DELETE" and event.memory_id:
            self.store.delete_fact(event.memory_id)
        # NONE -> nothing to do
        logger.info("Memory %s", event.describe())
        return event

    # ==================================================================
    # The public entry point: run both phases
    # ==================================================================
    def learn(
        self,
        new_messages: list[dict],
        user_id: str,
        *,
        summary: str = "",
        recent_messages: list[dict] | None = None,
        refresh_summary: bool = False,
    ) -> LearnResult:
        """
        Learn from the newest messages of a conversation.

        new_messages    : usually [user message, assistant reply] of the last turn
        summary         : the rolling summary so far (may be empty)
        recent_messages : a few earlier messages, only for resolving references
        refresh_summary : True every N turns (see SUMMARY_REFRESH_EVERY)
        """
        result = LearnResult(summary=summary)

        # Phase 1a - summary (optional, every N turns)
        if refresh_summary:
            result.summary = self.summarize(summary, (recent_messages or []) + new_messages)
            result.summary_refreshed = True

        # Phase 1b - extraction
        result.facts = self.extract_facts(new_messages, summary=result.summary, recent_messages=recent_messages)

        # Phase 2 - decisions + apply
        result.events = self.decide_and_apply(result.facts, user_id)
        return result

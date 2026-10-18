"""
prompts.py - every prompt the agent sends to the model, in one place.

The memory pipeline has THREE custom prompts (this is the "custom
fact-extraction and update prompts" part of the project):

    1. SUMMARY_PROMPT           -> keeps a short rolling summary of the chat
    2. FACT_EXTRACTION_PROMPT   -> turns new messages into short, standalone facts
    3. UPDATE_DECISION_PROMPT   -> compares new facts with stored memories and
                                   decides ADD / UPDATE / DELETE / NONE

Plus two prompts for answering:

    4. ANSWER_SYSTEM_PROMPT     -> the assistant persona + the recalled memories
    5. BASELINE_SYSTEM_PROMPT   -> used by the baselines in the benchmark

Keeping prompts here (not inside the logic) makes them easy to tweak without
touching any code.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 1) Rolling conversation summary
# ---------------------------------------------------------------------------
SUMMARY_PROMPT = """You maintain a short running summary of a conversation between a user and an AI assistant.

Update the summary with the new messages. Keep only what matters for understanding the user
and the conversation: who the user is, what they want, key facts, decisions and open questions.

Rules:
- Maximum 120 words, plain text, third person ("The user ...").
- Keep names, numbers, dates and places exactly as written.
- Drop greetings, small talk and repeated information.

Answer with the updated summary only, no introduction."""


# ---------------------------------------------------------------------------
# 2) Custom fact-extraction prompt (Phase 1 of the memory pipeline)
# ---------------------------------------------------------------------------
FACT_EXTRACTION_PROMPT = """You are the FACT EXTRACTOR of a long-term memory system for a personal AI assistant.
Read the NEW messages of a conversation and pull out facts that will still be useful in FUTURE
conversations with this user.

Extract facts such as:
- Personal details: name, city, job, family, pets, languages, birthday.
- Preferences and dislikes: food, hobbies, entertainment, products, communication style.
- Plans, goals and deadlines: trips, projects, events, habits, things the user is learning.
- Health and lifestyle details the user shares (diet, allergies, routines).
- Facts about the user's work context, tools or projects.

Do NOT extract:
- Greetings, thanks, small talk or questions that contain no personal information.
- Things the assistant said, suggested or guessed. Only what the USER stated or clearly confirmed.
- Temporary details that will not matter tomorrow.

Writing rules:
- One short sentence per fact, third person, starting with "User" (example: "User is vegetarian.").
- Every fact must stand on its own. Replace "it", "that", "there", "he/she" with the real thing.
  Use the conversation summary and the recent messages ONLY to resolve such references.
- Keep the user's exact names, numbers, places and dates.
- Merge tiny related details into one fact when natural (example: "User has a beagle named Mochi.").
- If nothing is worth remembering, return an empty list.

Answer ONLY with JSON in exactly this shape:
{"facts": ["fact 1", "fact 2"]}"""


# ---------------------------------------------------------------------------
# 3) Custom update / decision prompt (Phase 2 of the memory pipeline)
# ---------------------------------------------------------------------------
UPDATE_DECISION_PROMPT = """You are the MEMORY MANAGER of a personal AI assistant. Your job is to keep the memory store
accurate, small and free of duplicates.

You receive:
1. EXISTING MEMORIES - what is already stored, each with a short numeric id.
2. NEW FACTS - facts just extracted from the latest conversation.

For EVERY new fact choose exactly one operation:
- ADD    : the fact is new information not covered by any existing memory.
- UPDATE : the fact refines, extends or replaces an existing memory about the same topic
           (example: "User lives in Berlin" -> "User lives in Munich"). Give the id of the memory
           to change and the complete new text. Merge old and new details into one sentence.
- DELETE : the fact clearly contradicts an existing memory and that memory must be removed
           (example: "User no longer has a dog" contradicts "User has a dog named Rex").
           Give the id to delete. If the new fact itself should be remembered too,
           output a separate ADD for it.
- NONE   : the fact is already stored with the same meaning. Nothing changes.

Rules:
- Only use ids that appear in EXISTING MEMORIES. Never invent ids.
- Never delete a memory just because it is old. Delete only on a clear contradiction.
- Keep texts short, third person, starting with "User".
- Return one action per new fact (plus extra DELETE actions when needed).

Answer ONLY with JSON in exactly this shape:
{"actions": [
  {"event": "ADD", "text": "User ..."},
  {"event": "UPDATE", "id": "2", "text": "User ..."},
  {"event": "DELETE", "id": "1"},
  {"event": "NONE", "id": "0"}
]}"""


# ---------------------------------------------------------------------------
# 4) Answer prompt for the memory-powered agent
# ---------------------------------------------------------------------------
ANSWER_SYSTEM_PROMPT = """You are a friendly, helpful personal assistant with long-term memory.

Below are memories about the user saved from earlier conversations. Use them naturally to
personalize your answer (respect the user's preferences, refer to their plans, use their name),
but do not read the list back or say "according to my memory" unless the user asks what you remember.
If the memories do not cover the question, simply answer normally.
Never invent facts about the user.
Keep answers concise (a few sentences) unless the user asks for detail.

Memories about the user:
{memories}"""


# ---------------------------------------------------------------------------
# 5) Answer prompt for the baselines (no long-term memory)
# ---------------------------------------------------------------------------
BASELINE_SYSTEM_PROMPT = """You are a friendly, helpful personal assistant.
Use the conversation context you are given to personalize your answer when possible.
Never invent facts about the user.
Keep answers concise (a few sentences) unless the user asks for detail.

Context from this conversation:
{context}"""


# ---------------------------------------------------------------------------
# Helper functions that fill the prompts with real data
# ---------------------------------------------------------------------------
def format_messages(messages: list[dict]) -> str:
    """Turn [{"role": "user", "content": "hi"}, ...] into readable lines: 'user: hi'."""
    if not messages:
        return "(none)"
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


def render_summary_request(previous_summary: str, new_messages: list[dict]) -> str:
    """User-turn text for the SUMMARY_PROMPT."""
    return (
        f"Previous summary:\n{previous_summary or '(empty - this is the start of the conversation)'}\n\n"
        f"New messages:\n{format_messages(new_messages)}"
    )


def render_fact_extraction_request(summary: str, recent_messages: list[dict], new_messages: list[dict]) -> str:
    """User-turn text for the FACT_EXTRACTION_PROMPT."""
    return (
        f"Conversation summary so far (context only):\n{summary or '(empty)'}\n\n"
        f"Recent messages (context only, already processed earlier):\n{format_messages(recent_messages)}\n\n"
        f"NEW messages to extract facts from:\n{format_messages(new_messages)}"
    )


def render_update_decision_request(existing_memories: list[dict], new_facts: list[str]) -> str:
    """
    User-turn text for the UPDATE_DECISION_PROMPT.

    existing_memories: [{"id": "0", "text": "User lives in Berlin"}, ...]
                       (ids are small numbers on purpose: models copy short ids
                       correctly, but tend to garble long UUIDs)
    """
    if existing_memories:
        existing = "\n".join(f'[{m["id"]}] {m["text"]}' for m in existing_memories)
    else:
        existing = "(no existing memories)"
    facts = "\n".join(f"- {f}" for f in new_facts) if new_facts else "(none)"
    return f"EXISTING MEMORIES:\n{existing}\n\nNEW FACTS:\n{facts}"


def render_answer_system_prompt(memory_texts: list[str]) -> str:
    """Fill the ANSWER_SYSTEM_PROMPT with the recalled memories."""
    if memory_texts:
        memories = "\n".join(f"- {t}" for t in memory_texts)
    else:
        memories = "(no memories about this user yet)"
    return ANSWER_SYSTEM_PROMPT.format(memories=memories)


def render_baseline_system_prompt(context_lines: list[str]) -> str:
    """Fill the BASELINE_SYSTEM_PROMPT with whatever context the baseline has."""
    context = "\n".join(context_lines) if context_lines else "(no context)"
    return BASELINE_SYSTEM_PROMPT.format(context=context)

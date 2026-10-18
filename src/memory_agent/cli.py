"""
cli.py - chat with the agent from the terminal.

Run it with:   python scripts/chat.py           (or press F5 in VS Code -> "Chat (CLI)")

Commands you can type at the prompt:
    /memories        show everything the agent remembers about you
    /forget          delete all memories of the current user
    /user <id>       switch to another user (each user has separate memories)
    /stats           tokens, cost and latency so far
    /help            this list
    /exit            quit (background learning is finished first)
"""

from __future__ import annotations

import sys

from memory_agent.agent import MemoryAgent
from memory_agent.settings import configure_logging, get_settings

HELP = __doc__.split("Commands you can type at the prompt:")[1]


def print_turn_details(result) -> None:
    """Small grey line under each answer showing what the memory layer did."""
    recalled = f"recalled {len(result.recalled)} memor{'y' if len(result.recalled) == 1 else 'ies'}"
    parts = [recalled, f"{result.prompt_tokens} prompt tokens", f"{result.answer_seconds:.2f}s"]
    if result.learn is not None:
        changes = [e.describe() for e in result.learn.events if e.event != "NONE"]
        parts.append("learned: " + ("; ".join(changes) if changes else "nothing new"))
    else:
        parts.append("learning in background")
    print(f"   ↳ {' | '.join(parts)}")


def main() -> None:
    settings = get_settings()
    settings.validate()
    configure_logging(settings)

    agent = MemoryAgent(settings)
    if not agent.store.ping():
        print("Qdrant is not reachable. Start it with:  docker compose up -d qdrant")
        sys.exit(1)

    session = agent.start_session(settings.default_user_id)
    mode = "local Ollama" if settings.is_local else "OpenAI"
    print(f"\n🧠 Self-Learning Memory Agent  (user: {session.user_id}, model: {settings.llm_model} via {mode})")
    print(f"   {agent.store.count(session.user_id)} memories stored for this user. Type /help for commands.\n")

    try:
        while True:
            try:
                text = input("You: ").strip()
            except EOFError:
                break
            if not text:
                continue

            # ----------------------------- slash commands -----------------------------
            if text.lower() in ("/exit", "/quit", "exit", "quit"):
                break
            if text.lower() == "/help":
                print(HELP)
                continue
            if text.lower() == "/memories":
                memories = agent.list_memories(session.user_id)
                print(f"\n{len(memories)} memories for {session.user_id}:")
                for m in memories:
                    print(f"  • {m.text}")
                print()
                continue
            if text.lower() == "/forget":
                agent.forget_user(session.user_id)
                session = agent.start_session(session.user_id)
                print("All memories deleted for this user.\n")
                continue
            if text.lower().startswith("/user "):
                agent.wait_for_learning()
                session = agent.start_session(text.split(maxsplit=1)[1].strip())
                print(f"Switched to user '{session.user_id}' ({agent.store.count(session.user_id)} memories).\n")
                continue
            if text.lower() == "/stats":
                s = agent.tracker.summary()
                print(f"\ncalls={s['total_calls']} prompt_tokens={s['total_prompt_tokens']} "
                      f"completion_tokens={s['total_completion_tokens']} est_cost=${s['total_cost_usd']:.5f}")
                for purpose, row in s["per_purpose"].items():
                    print(f"  {purpose:<10} calls={row['calls']:<4} prompt={row['prompt_tokens']:<7} "
                          f"completion={row['completion_tokens']:<6} time={row['seconds']:.2f}s")
                print()
                continue

            # ----------------------------- a normal chat turn -----------------------------
            result = agent.chat(session, text)
            print(f"\nAI: {result.reply}")
            print_turn_details(result)
            print()
    except KeyboardInterrupt:
        pass
    finally:
        print("\nFinishing background learning ... ", end="", flush=True)
        agent.close()
        print("bye!")


if __name__ == "__main__":
    main()

"""
check_setup.py - run this FIRST. It tells you what is missing before you start chatting.

    python scripts/check_setup.py

Checks:
  1. .env file exists and the required keys are set
  2. Qdrant answers on QDRANT_HOST:QDRANT_PORT
  3. the LLM answers one tiny test message
  4. the embedder returns a vector of the configured size (EMBED_DIMS)
"""

import _bootstrap  # noqa: F401

import sys

from memory_agent.settings import PROJECT_ROOT, configure_logging, get_settings

OK, BAD = "✅", "❌"


def main() -> int:
    problems = 0

    # 1) .env + settings -----------------------------------------------------
    env_file = PROJECT_ROOT / ".env"
    print(f"{OK if env_file.exists() else BAD} .env file: {env_file}")
    if not env_file.exists():
        print("   -> copy .env.example to .env and fill in OPENAI_API_KEY (or switch to Ollama)")
        problems += 1

    settings = get_settings()
    configure_logging(settings)
    try:
        settings.validate()
        print(f"{OK} settings valid  (LLM: {settings.llm_provider}/{settings.llm_model}, "
              f"embedder: {settings.embedder_provider}/{settings.embed_model} x{settings.embed_dims}, "
              f"pipeline: {settings.memory_pipeline})")
    except ValueError as exc:
        print(f"{BAD} settings: {exc}")
        return 1

    # 2) Qdrant -------------------------------------------------------------
    from memory_agent.memory_store import MemoryStore

    try:
        store = MemoryStore(settings)
        assert store.ping()
        print(f"{OK} Qdrant reachable at {settings.qdrant_host}:{settings.qdrant_port} "
              f"(collection '{settings.qdrant_collection}', {store.count()} memories)")
    except Exception as exc:  # noqa: BLE001
        print(f"{BAD} Qdrant: {exc}")
        print("   -> start it with:  docker compose up -d qdrant")
        return 1

    # 3) LLM ----------------------------------------------------------------
    from memory_agent.llm import LLMClient

    try:
        llm = LLMClient(settings)
        reply = llm.chat([{"role": "user", "content": "Reply with the single word: ready"}], max_tokens=5)
        print(f"{OK} LLM answered: {reply.text!r}  ({reply.prompt_tokens}+{reply.completion_tokens} tokens, {reply.latency_seconds:.2f}s)")
    except Exception as exc:  # noqa: BLE001
        print(f"{BAD} LLM call failed: {exc}")
        problems += 1

    # 4) Embedder -----------------------------------------------------------
    try:
        vector = store.memory.embedding_model.embed("hello world", "search")
        if len(vector) == settings.embed_dims:
            print(f"{OK} embedder returned a vector of size {len(vector)}")
        else:
            print(f"{BAD} embedder returned size {len(vector)} but EMBED_DIMS={settings.embed_dims}. "
                  f"Fix EMBED_DIMS in .env and use a NEW QDRANT_COLLECTION name.")
            problems += 1
    except Exception as exc:  # noqa: BLE001
        print(f"{BAD} embedding call failed: {exc}")
        problems += 1

    print("\nAll good! Run:  python scripts/chat.py" if problems == 0 else f"\n{problems} problem(s) found - see above.")
    return 0 if problems == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

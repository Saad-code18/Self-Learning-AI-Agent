"""
inspect_memories.py - print everything stored for a user (and the change history of each memory).

    python scripts/inspect_memories.py                 # uses DEFAULT_USER_ID
    python scripts/inspect_memories.py alice           # a specific user
    python scripts/inspect_memories.py alice --history # also show ADD/UPDATE/DELETE history
"""

import _bootstrap  # noqa: F401

import argparse

from memory_agent.memory_store import MemoryStore
from memory_agent.settings import configure_logging, get_settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("user_id", nargs="?", help="user whose memories to show (default: DEFAULT_USER_ID)")
    parser.add_argument("--history", action="store_true", help="also print the audit trail of every memory")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings)
    store = MemoryStore(settings)
    user_id = args.user_id or settings.default_user_id

    memories = store.list_all(user_id)
    print(f"\n{len(memories)} memories for user '{user_id}'  (total in collection: {store.count()})\n")
    for m in memories:
        print(f"• {m.text}")
        print(f"    id={m.id}  created={m.created_at}  updated={m.updated_at}")
        if args.history:
            for h in store.history(m.id):
                old, new = h.get("old_memory"), h.get("new_memory")
                print(f"    {h.get('event'):<7} {old!r} -> {new!r}  at {h.get('created_at')}")


if __name__ == "__main__":
    main()

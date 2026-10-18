"""
reset_memories.py - delete memories.

    python scripts/reset_memories.py alice          # forget one user
    python scripts/reset_memories.py --all --yes    # wipe the whole collection (every user)
"""

import _bootstrap  # noqa: F401

import argparse

from memory_agent.memory_store import MemoryStore
from memory_agent.settings import configure_logging, get_settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("user_id", nargs="?", help="user to forget (default: DEFAULT_USER_ID)")
    parser.add_argument("--all", action="store_true", help="delete EVERY memory of EVERY user")
    parser.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings)
    store = MemoryStore(settings)

    if args.all:
        total = store.count()
        if not args.yes and input(f"Delete ALL {total} memories in '{settings.qdrant_collection}'? [y/N] ").lower() != "y":
            print("Cancelled.")
            return
        # Mem0's reset() drops the collection AND the history table, then recreates the collection.
        store.memory.reset()
        print(f"Deleted {total} memories. Collection is empty now.")
        return

    user_id = args.user_id or settings.default_user_id
    before = store.count(user_id)
    store.delete_all(user_id)
    print(f"Deleted {before} memories for user '{user_id}'.")


if __name__ == "__main__":
    main()

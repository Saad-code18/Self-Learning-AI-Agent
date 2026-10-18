"""
chat.py - start the interactive chat (same as pressing F5 -> "Chat (CLI)" in VS Code).

    python scripts/chat.py
"""

import _bootstrap  # noqa: F401  (adds src/ to the import path)

from memory_agent.cli import main

if __name__ == "__main__":
    main()

"""
_bootstrap.py - makes `import memory_agent` work no matter where you run a script from.

Every script starts with `import _bootstrap  # noqa` which adds the `src/`
folder to Python's import path. That way you never need to set PYTHONPATH
or install the package: `python scripts/chat.py` just works.
"""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

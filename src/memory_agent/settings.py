"""
settings.py - ONE place for all configuration.

Why this file exists:
    Every other module (the LLM client, the memory store, the CLI, the API...)
    needs the same values: which model to use, where Qdrant runs, how many
    memories to recall, etc.  Instead of reading environment variables all over
    the code base, we read them ONCE here and pass a `Settings` object around.

How it works:
    1. `python-dotenv` loads the `.env` file in the project root (if it exists).
    2. We read each variable with a sensible default so the project still runs
       with a minimal `.env` (only OPENAI_API_KEY is truly required for OpenAI).
    3. `get_settings()` returns a frozen (read-only) dataclass.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Project root = two folders above this file  (src/memory_agent/settings.py -> project root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load variables from `.env` into the process environment.
# `override=False` means: if a variable is ALREADY set in the real environment
# (for example by Docker Compose), the real environment wins.
load_dotenv(PROJECT_ROOT / ".env", override=False)

# Mem0 sends anonymous usage statistics to PostHog unless this is "false".
# Because this is a SELF-HOSTED, privacy-friendly stack we switch it off.
# IMPORTANT: this must happen BEFORE `import mem0` anywhere, because mem0 reads
# the variable at import time.  settings.py is imported first everywhere, so
# putting it here guarantees the right order.
os.environ.setdefault("MEM0_TELEMETRY", "false")


# ----------------------------------------------------------------------------
# Small helpers so every variable is parsed the same way
# ----------------------------------------------------------------------------
def _env(name: str, default: str | None = None) -> str | None:
    """Read a text variable. Empty strings count as 'not set'."""
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def _env_int(name: str, default: int) -> int:
    """Read a whole number variable (falls back to default on bad input)."""
    raw = _env(name)
    try:
        return int(raw) if raw is not None else default
    except ValueError:
        logging.warning("Environment variable %s=%r is not a number, using %s", name, raw, default)
        return default


def _env_float(name: str, default: float) -> float:
    """Read a decimal number variable."""
    raw = _env(name)
    try:
        return float(raw) if raw is not None else default
    except ValueError:
        logging.warning("Environment variable %s=%r is not a number, using %s", name, raw, default)
        return default


def _env_bool(name: str, default: bool) -> bool:
    """Read a true/false variable. Accepts 1/0, true/false, yes/no, on/off."""
    raw = _env(name)
    if raw is None:
        return default
    return raw.lower() in ("1", "true", "yes", "on")


# ----------------------------------------------------------------------------
# The settings object
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Settings:
    """All knobs of the project. Frozen = cannot be changed by accident at runtime."""

    # --- Which AI provider answers questions and runs the memory pipeline ---
    # "openai" = OpenAI cloud API,  "ollama" = fully local models via Ollama
    llm_provider: str
    llm_model: str
    openai_api_key: str | None
    openai_base_url: str | None  # optional: OpenAI-compatible endpoint (Azure, vLLM, ...)
    ollama_base_url: str

    # --- Embeddings (turn text into vectors so Qdrant can search by meaning) ---
    embedder_provider: str  # "openai" or "ollama"
    embed_model: str
    embed_dims: int  # must match the model: 1536 for text-embedding-3-small, 768 for nomic-embed-text

    # --- Qdrant vector database (runs in Docker, see docker-compose.yml) ---
    qdrant_host: str
    qdrant_port: int
    qdrant_collection: str

    # --- Mem0 keeps an audit trail of every ADD/UPDATE/DELETE in a small SQLite file ---
    history_db_path: str

    # --- Memory pipeline behaviour ---
    memory_pipeline: str  # "custom" = our two-phase pipeline,  "mem0" = Mem0's built-in extraction
    recall_top_k: int  # how many memories to inject into the answer prompt
    similar_top_k: int  # how many existing memories to compare each new fact against
    summary_refresh_every: int  # refresh the rolling conversation summary every N turns
    recent_messages_window: int  # how many recent messages go into the answer prompt
    learn_in_background: bool  # run the learning step in a thread so replies are not delayed

    # --- Cost estimation (USD per 1 million tokens). Update from your provider's price page ---
    price_input_per_1m: float
    price_output_per_1m: float

    # --- Misc ---
    log_level: str
    default_user_id: str

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------
    @property
    def is_local(self) -> bool:
        """True when everything runs on your own machine (Ollama)."""
        return self.llm_provider == "ollama"

    def validate(self) -> None:
        """Fail fast with a clear message instead of a confusing stack trace later."""
        if self.llm_provider not in ("openai", "ollama"):
            raise ValueError(f"LLM_PROVIDER must be 'openai' or 'ollama', got {self.llm_provider!r}")
        if self.embedder_provider not in ("openai", "ollama"):
            raise ValueError(f"EMBEDDER_PROVIDER must be 'openai' or 'ollama', got {self.embedder_provider!r}")
        if self.memory_pipeline not in ("custom", "mem0"):
            raise ValueError(f"MEMORY_PIPELINE must be 'custom' or 'mem0', got {self.memory_pipeline!r}")
        needs_openai_key = self.llm_provider == "openai" or self.embedder_provider == "openai"
        if needs_openai_key and not self.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is missing. Copy `.env.example` to `.env` and paste your key, "
                "or switch to local models with LLM_PROVIDER=ollama and EMBEDDER_PROVIDER=ollama."
            )
        if self.embed_dims <= 0:
            raise ValueError("EMBED_DIMS must be a positive number (1536 for text-embedding-3-small).")


def get_settings() -> Settings:
    """Build the Settings object from environment variables (with defaults)."""
    llm_provider = (_env("LLM_PROVIDER", "openai") or "openai").lower()

    # Default models depend on the provider, so the project works out of the box for both.
    default_llm_model = "gpt-4o-mini" if llm_provider == "openai" else "llama3.1:8b"

    # The embedder normally uses the same provider as the LLM, but you can mix
    # (e.g. local Ollama LLM + OpenAI embeddings) by setting EMBEDDER_PROVIDER.
    embedder_provider = (_env("EMBEDDER_PROVIDER", llm_provider) or llm_provider).lower()
    default_embed_model = "text-embedding-3-small" if embedder_provider == "openai" else "nomic-embed-text"
    default_embed_dims = 1536 if embedder_provider == "openai" else 768

    history_path = Path(_env("HISTORY_DB_PATH", str(PROJECT_ROOT / "data" / "mem0_history.db")) or "").expanduser()
    if not history_path.is_absolute():
        # A relative path (like ./data/mem0_history.db) is taken relative to the PROJECT folder,
        # so it does not matter from which directory you start a script.
        history_path = PROJECT_ROOT / history_path
    history_path = history_path.resolve()
    # Make sure the folder for the SQLite history file exists (mem0 does not create it).
    history_path.parent.mkdir(parents=True, exist_ok=True)

    settings = Settings(
        llm_provider=llm_provider,
        llm_model=_env("LLM_MODEL", default_llm_model) or default_llm_model,
        openai_api_key=_env("OPENAI_API_KEY"),
        openai_base_url=_env("OPENAI_BASE_URL"),
        ollama_base_url=(_env("OLLAMA_BASE_URL", "http://localhost:11434") or "http://localhost:11434").rstrip("/"),
        embedder_provider=embedder_provider,
        embed_model=_env("EMBED_MODEL", default_embed_model) or default_embed_model,
        embed_dims=_env_int("EMBED_DIMS", default_embed_dims),
        qdrant_host=_env("QDRANT_HOST", "localhost") or "localhost",
        qdrant_port=_env_int("QDRANT_PORT", 6333),
        qdrant_collection=_env("QDRANT_COLLECTION", "agent_memories") or "agent_memories",
        history_db_path=str(history_path),
        memory_pipeline=(_env("MEMORY_PIPELINE", "custom") or "custom").lower(),
        recall_top_k=_env_int("RECALL_TOP_K", 5),
        similar_top_k=_env_int("SIMILAR_TOP_K", 5),
        summary_refresh_every=max(1, _env_int("SUMMARY_REFRESH_EVERY", 3)),
        recent_messages_window=_env_int("RECENT_MESSAGES_WINDOW", 6),
        learn_in_background=_env_bool("LEARN_IN_BACKGROUND", False),
        price_input_per_1m=_env_float("PRICE_INPUT_PER_1M_TOKENS", 0.15),
        price_output_per_1m=_env_float("PRICE_OUTPUT_PER_1M_TOKENS", 0.60),
        log_level=(_env("LOG_LEVEL", "INFO") or "INFO").upper(),
        default_user_id=_env("DEFAULT_USER_ID", "demo_user") or "demo_user",
    )
    return settings


def configure_logging(settings: Settings) -> None:
    """Set up readable log output and silence noisy third-party loggers."""
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # mem0 prints warnings about optional extras (spaCy / fastembed) that we do
    # not need.  httpx logs every request.  Keep the console clean unless the
    # user explicitly asks for DEBUG output.
    if settings.log_level != "DEBUG":
        for noisy in ("mem0", "httpx", "httpx2", "httpcore", "openai", "posthog"):
            logging.getLogger(noisy).setLevel(logging.ERROR)

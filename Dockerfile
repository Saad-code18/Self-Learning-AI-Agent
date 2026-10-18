# ---------------------------------------------------------------------------
# Dockerfile - packages the agent (FastAPI service) into an image.
#
# Build & run together with Qdrant:   docker compose --profile app up --build
# The image does NOT contain your .env; Compose passes it in at runtime.
# ---------------------------------------------------------------------------
FROM python:3.12-slim

# Do not write .pyc files, and print logs immediately (no buffering).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    MEM0_TELEMETRY=false

WORKDIR /app

# Install dependencies first (this layer is cached until requirements.txt changes).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the application code.
COPY src ./src
COPY scripts ./scripts
COPY data ./data

# Mem0's history database and other runtime files live here (a volume in compose).
RUN mkdir -p /data
ENV HISTORY_DB_PATH=/data/mem0_history.db

EXPOSE 8000
# Default command: the HTTP API. Override for the CLI, e.g.
#   docker compose --profile app run --rm app python scripts/chat.py
CMD ["uvicorn", "memory_agent.api:app", "--host", "0.0.0.0", "--port", "8000"]

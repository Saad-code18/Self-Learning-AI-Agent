# 🧠 Self-Learning Conversational AI Agent with Long-Term Memory

A chat agent that **remembers you across sessions**. After every turn it distils the
conversation into short facts, decides whether each fact should be **added, updated or
deleted** in its memory, and stores the result in a **self-hosted Mem0 + Qdrant** stack
running in Docker. On the next conversation (even after a restart) it recalls the
relevant facts and personalises its answers.

Inspired by Dave Ebbelaar's video *How to Build Self-Learning AI Agents*
(https://www.youtube.com/watch?v=ynhl8KjjS3Y) and the Mem0 paper's two-phase memory design,
rebuilt for the current `mem0ai` 2.x API and extended with a custom pipeline, baselines and a benchmark.

```
┌────────────┐   1. recall (vector search)   ┌──────────────────────────┐
│   user     │ ───────────────────────────▶  │  Qdrant  (Docker)        │
│  message   │                                │  distilled facts + vectors│
└─────┬──────┘                                └──────────▲───────────────┘
      │ 2. answer = persona + recalled memories           │ 4. ADD / UPDATE / DELETE
      ▼                                                   │
┌────────────┐   3. learn: summarise → extract facts → decide ──┘
│    LLM     │      (OpenAI  or  fully local Ollama)
└────────────┘
```

---

## Quick start (about 5 minutes)

**You need:** Python 3.10+, Docker Desktop, and an OpenAI API key (or Ollama for a 100 % local setup).

1. **Open the folder in VS Code** (`File → Open Folder…`). Accept the recommended extensions.
2. **Create your `.env`:** copy `.env.example` to `.env` and paste your key into `OPENAI_API_KEY=`.
3. **Start Qdrant** (the memory database) in a terminal:
   ```bash
   docker compose up -d
   ```
4. **Create a virtual environment and install the requirements:**
   ```bash
   python -m venv .venv
   # Windows:  .venv\Scripts\activate      macOS/Linux:  source .venv/bin/activate
   pip install -r requirements.txt
   ```
   (VS Code will ask "use this environment?" – say yes, or pick `.venv` via `Python: Select Interpreter`.)
5. **Check everything is wired up:**
   ```bash
   python scripts/check_setup.py
   ```
6. **Chat!** Press **F5** and choose **"Chat (CLI)"**, or run `python scripts/chat.py`.

```
🧠 Self-Learning Memory Agent  (user: demo_user, model: gpt-4o-mini via OpenAI)
You: Hi, I'm Priya. I live in Toronto and I'm vegetarian.
AI: Nice to meet you, Priya! ...
   ↳ recalled 0 memories | 142 prompt tokens | 1.10s | learned: ADD 'User's name is Priya.'; ADD 'User lives in Toronto.'; ADD 'User is vegetarian.'
You: /exit
```

Start the chat again (new session) and ask *"What should I cook tonight?"* – the agent
will suggest something vegetarian because the facts survived in Qdrant.

Useful commands inside the chat: `/memories`, `/forget`, `/user <id>`, `/stats`, `/help`, `/exit`.

---

## What's in the box

| Path | What it does |
|------|--------------|
| `src/memory_agent/settings.py` | Reads `.env` once; every knob of the project lives here |
| `src/memory_agent/llm.py` | One LLM client for OpenAI **or** Ollama, with token/cost/latency tracking |
| `src/memory_agent/prompts.py` | The custom **summary**, **fact-extraction** and **update-decision** prompts |
| `src/memory_agent/memory_store.py` | Mem0 + Qdrant wrapper: `add_fact / update_fact / delete_fact / search / count / history` |
| `src/memory_agent/pipeline.py` | The **two-phase memory pipeline** (see below) |
| `src/memory_agent/agent.py` | The chat agent: recall → answer → learn (optionally in a background thread) |
| `src/memory_agent/baseline.py` | Stateless-RAG and full-dialogue baselines used by the benchmark |
| `src/memory_agent/cli.py` | Terminal chat UI |
| `src/memory_agent/api.py` | FastAPI service (`/chat`, `/memories/{user}`, `/health`, `/stats`) |
| `scripts/chat.py` | Start the chat |
| `scripts/check_setup.py` | Verifies `.env`, Qdrant, the LLM and the embedder |
| `scripts/seed_memories.py` | Inserts **1,100** synthetic memories for 50 users (scale test) |
| `scripts/benchmark.py` | Measures accuracy, prompt tokens, cost and latency vs the baselines |
| `scripts/inspect_memories.py` / `reset_memories.py` | Look at / delete stored memories (incl. the ADD/UPDATE/DELETE history) |
| `data/eval_dataset.json` | Multi-session evaluation conversations + questions |
| `tests/` | 28 unit tests (no API key needed) + 2 integration tests against real Qdrant |
| `docker-compose.yml`, `Dockerfile` | Qdrant (always), the API (`--profile app`), Ollama (`--profile local`) |
| `.vscode/` | F5 launch configs and tasks |

---

## How the memory pipeline works

After each turn (`user message` + `assistant reply`) the agent runs `MemoryPipeline.learn()`:

**Phase 1 – Extraction**
1. *Summarise* (every `SUMMARY_REFRESH_EVERY` turns): a ≤120-word rolling summary of the session
   gives the extractor context to resolve "it / there / he".
2. *Extract facts*: the custom `FACT_EXTRACTION_PROMPT` turns the new messages into short,
   standalone, third-person facts – `{"facts": ["User is vegetarian.", "User lives in Toronto."]}`.
   Small talk and assistant guesses are ignored.

**Phase 2 – Update / decisioning**
3. For every fact, the most similar existing memories are fetched from Qdrant.
4. The custom `UPDATE_DECISION_PROMPT` sees *existing memories* (with short ids `0, 1, 2…` to avoid
   hallucinated UUIDs) and the *new facts*, and returns one action per fact:
   `ADD`, `UPDATE <id>` (e.g. *Berlin → Munich*), `DELETE <id>` (contradiction) or `NONE` (already known).
5. The actions are applied through Mem0 (`add(infer=False)`, `update`, `delete`). Mem0 records every
   change in a SQLite history table, so `inspect_memories.py --history` shows the full audit trail.

Safety nets: bad JSON → no changes; unknown ids → `UPDATE` becomes `ADD`, `DELETE` is ignored;
an empty store skips the decision call entirely (one LLM call less).

Only the distilled facts are stored – never the full dialogue – which is why the answer prompt stays
a few hundred tokens no matter how long the user has been talking to the agent.

`MEMORY_PIPELINE=mem0` switches to Mem0's built-in extraction (the approach in the video) if you
want to compare the two.

---

## Configuration (`.env`)

| Variable | Default | Meaning |
|----------|---------|---------|
| `LLM_PROVIDER` | `openai` | `openai` or `ollama` |
| `LLM_MODEL` | `gpt-4o-mini` / `llama3.1:8b` | chat model for answers and the pipeline |
| `OPENAI_API_KEY` | – | required for OpenAI |
| `EMBEDDER_PROVIDER`, `EMBED_MODEL`, `EMBED_DIMS` | follow the LLM provider | embedding model; **dims must match the model** (1536 for `text-embedding-3-small`, 768 for `nomic-embed-text`) |
| `QDRANT_HOST`, `QDRANT_PORT`, `QDRANT_COLLECTION` | `localhost`, `6333`, `agent_memories` | change the collection name whenever you change the embedding model |
| `MEMORY_PIPELINE` | `custom` | `custom` (two-phase pipeline) or `mem0` |
| `RECALL_TOP_K` | `5` | memories injected per answer |
| `SIMILAR_TOP_K` | `5` | existing memories compared to each new fact |
| `SUMMARY_REFRESH_EVERY` | `3` | turns between summary refreshes |
| `RECENT_MESSAGES_WINDOW` | `6` | messages of the current session kept in the prompt |
| `LEARN_IN_BACKGROUND` | `false` | `true` = learn in a thread; the user never waits for the pipeline |
| `PRICE_INPUT_PER_1M_TOKENS`, `PRICE_OUTPUT_PER_1M_TOKENS` | `0.15`, `0.60` | for cost estimates – edit to your provider's prices |

### 100 % local (no API key, nothing leaves your machine)

```bash
# Option A – Ollama installed on your computer (https://ollama.com)
ollama pull llama3.1:8b && ollama pull nomic-embed-text
# Option B – Ollama inside Docker
docker compose --profile local up -d
docker compose exec ollama ollama pull llama3.1:8b
docker compose exec ollama ollama pull nomic-embed-text
```
Then in `.env`: `LLM_PROVIDER=ollama`, `LLM_MODEL=llama3.1:8b`, `EMBEDDER_PROVIDER=ollama`,
`EMBED_MODEL=nomic-embed-text`, `EMBED_DIMS=768`, `QDRANT_COLLECTION=agent_memories_local`.
Small local models follow the JSON prompts less reliably than GPT-class models; if extraction looks
poor, try a larger model (e.g. `qwen2.5:14b`).

---

## Running the whole stack in Docker

```bash
docker compose --profile app up --build      # Qdrant + the agent's HTTP API
```
Then open http://localhost:8000/docs (Swagger UI) or:
```bash
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" \
     -d '{"user_id": "priya", "session_id": "s1", "message": "I moved to Denver last month"}'
curl http://localhost:8000/memories/priya
```
The CLI also works inside the container: `docker compose --profile app run --rm app python scripts/chat.py`.
Qdrant's own dashboard is at http://localhost:6333/dashboard.

---

## Benchmark: what the memory layer buys you

```bash
python scripts/seed_memories.py      # optional: 1,100 memories so search runs at realistic scale
python scripts/benchmark.py          # add --judge to let the LLM grade answers as well
```

Three agents replay the same multi-session conversations from `data/eval_dataset.json`, then answer
questions in a **new** session:

| agent | what it is | shows |
|-------|------------|-------|
| `memory` | this project (Mem0 + Qdrant, distilled facts) | – |
| `stateless` | stateless RAG: retrieval over the current session only, nothing across sessions | **accuracy** gap |
| `fullhistory` | the naive alternative: the entire dialogue of all sessions in every prompt | **token / cost / latency** gap |

The script prints a table plus the headline deltas (personalisation accuracy vs *stateless*,
prompt-token / cost / latency reduction vs *fullhistory*, memory-search latency at the current
collection size) and saves everything to `data/benchmark_results.json`. LLM answers are random, so run
it 2–3 times and average before quoting numbers.

---

## Tests

```bash
python -m pytest -v                                  # 28 unit tests, fakes only – no key, no Qdrant
RUN_INTEGRATION=1 python -m pytest tests/test_integration_qdrant.py -v   # real Qdrant, fake embedder
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Qdrant is not reachable` | `docker compose up -d qdrant`, then `docker compose ps` should show it healthy |
| `OPENAI_API_KEY is missing` | copy `.env.example` → `.env` and paste the key, or switch to Ollama |
| Qdrant error about vector size / dimensions | `EMBED_DIMS` does not match the embedding model – fix it and use a **new** `QDRANT_COLLECTION` |
| `ModuleNotFoundError: memory_agent` | run scripts from the project root (`python scripts/...`) – they add `src/` to the path themselves |
| The agent "forgets" between runs | check `/memories`; memories live in the `qdrant_storage` Docker volume – `docker compose down -v` deletes them |
| Ollama in Docker cannot be reached by the app container | set `OLLAMA_BASE_URL_DOCKER=http://ollama:11434` (compose profile) or `http://host.docker.internal:11434` (host install) |

---

## Resume mapping

* **Persistent long-term memory (Mem0 + Qdrant), cross-session recall** → `memory_store.py`, `agent.py`, `docker-compose.yml`; measured by `benchmark.py` (`memory` vs `stateless`).
* **Two-phase pipeline: summarisation → fact extraction → add/update/delete decisioning, storing distilled facts** → `pipeline.py` + `prompts.py`; token/cost savings measured vs `fullhistory`.
* **Self-hosted, Dockerized stack with custom prompts, no vendor lock-in, 1,000+ memories** → `docker-compose.yml`, `Dockerfile`, Ollama option, `seed_memories.py`, memory-layer latency in the benchmark.

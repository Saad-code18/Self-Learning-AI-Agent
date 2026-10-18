"""
benchmark.py - measure what the memory layer actually buys you.

    python scripts/benchmark.py                 # all 3 agents, keyword grading
    python scripts/benchmark.py --judge         # also let the LLM grade the answers
    python scripts/benchmark.py --only memory stateless
    python scripts/benchmark.py --scenarios 2   # quick run on the first 2 scenarios

What it does
------------
For every scenario in data/eval_dataset.json and for each agent:
    1. replay the sessions (the user shares facts)             -> the agent learns
    2. open a NEW session and ask the questions                -> can it recall?
    3. grade the answers, count tokens, cost and latency

Agents compared
    memory       : our MemoryAgent (Mem0 + Qdrant, distilled facts)      <- the project
    stateless    : stateless RAG baseline (no cross-session memory)       <- accuracy comparison
    fullhistory  : full-dialogue-in-prompt baseline (remembers everything,
                   but the prompt grows with every message)               <- token/cost/latency comparison

The numbers printed at the end map directly to the three resume claims
(personalization accuracy, prompt-token / cost reduction, latency reduction).
Results are also saved to data/benchmark_results.json.

Cost: ~150-250 small LLM calls (a few cents with gpt-4o-mini, free with Ollama).
"""

import _bootstrap  # noqa: F401

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass, field

from memory_agent.agent import MemoryAgent
from memory_agent.baseline import FullHistoryBaseline, StatelessRAGBaseline
from memory_agent.llm import LLMClient, UsageTracker
from memory_agent.memory_store import MemoryStore
from memory_agent.settings import PROJECT_ROOT, configure_logging, get_settings

AGENT_KINDS = ("memory", "stateless", "fullhistory")
DATASET = PROJECT_ROOT / "data" / "eval_dataset.json"
RESULTS = PROJECT_ROOT / "data" / "benchmark_results.json"

JUDGE_PROMPT = """You grade whether an assistant's answer is consistent with a known fact about the user.
Fact about the user: {fact}
Question asked: {question}
Assistant answer: {answer}

Answer with exactly one word: YES if the answer correctly uses/reflects the fact, NO otherwise."""


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class QuestionResult:
    scenario: str
    question: str
    answer: str
    keyword_correct: bool
    judge_correct: bool | None
    prompt_tokens: int
    completion_tokens: int
    answer_seconds: float


@dataclass
class AgentReport:
    kind: str
    questions: list[QuestionResult] = field(default_factory=list)
    learn_seconds: list[float] = field(default_factory=list)  # per session-phase turn (memory agent only)
    answer_cost_usd: float = 0.0  # LLM cost of the answer calls in the question phase
    total_cost_usd: float = 0.0  # answer + learning cost in the question phase
    total_calls: int = 0

    # ---- derived metrics ----
    def accuracy(self, use_judge: bool = False) -> float:
        if not self.questions:
            return 0.0
        if use_judge:
            graded = [q.judge_correct for q in self.questions if q.judge_correct is not None]
            return 100.0 * sum(graded) / len(graded) if graded else 0.0
        return 100.0 * sum(q.keyword_correct for q in self.questions) / len(self.questions)

    def avg_prompt_tokens(self) -> float:
        return statistics.mean(q.prompt_tokens for q in self.questions) if self.questions else 0.0

    def avg_completion_tokens(self) -> float:
        return statistics.mean(q.completion_tokens for q in self.questions) if self.questions else 0.0

    def avg_answer_seconds(self) -> float:
        return statistics.mean(q.answer_seconds for q in self.questions) if self.questions else 0.0

    def avg_learn_seconds(self) -> float:
        return statistics.mean(self.learn_seconds) if self.learn_seconds else 0.0

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "accuracy_keywords_pct": round(self.accuracy(), 1),
            "accuracy_judge_pct": round(self.accuracy(use_judge=True), 1),
            "avg_prompt_tokens_per_query": round(self.avg_prompt_tokens(), 1),
            "avg_completion_tokens_per_query": round(self.avg_completion_tokens(), 1),
            "avg_answer_latency_s": round(self.avg_answer_seconds(), 3),
            "avg_learn_latency_s": round(self.avg_learn_seconds(), 3),
            "answer_cost_usd_question_phase": round(self.answer_cost_usd, 6),
            "total_cost_usd_question_phase": round(self.total_cost_usd, 6),
            "questions": [asdict(q) for q in self.questions],
        }


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------
def grade_keywords(answer: str, expected_groups: list[list[str]]) -> bool:
    """Correct when every group has at least one keyword present in the answer."""
    text = answer.lower()
    return all(any(kw.lower() in text for kw in group) for group in expected_groups)


def grade_with_judge(judge: LLMClient, fact: str, question: str, answer: str) -> bool:
    """Ask the model itself whether the answer reflects the fact (YES/NO)."""
    reply = judge.chat(
        [{"role": "user", "content": JUDGE_PROMPT.format(fact=fact, question=question, answer=answer)}],
        temperature=0.0,
        max_tokens=3,
        purpose="judge",
    )
    return reply.text.strip().upper().startswith("YES")


# ---------------------------------------------------------------------------
# Running one agent over the dataset
# ---------------------------------------------------------------------------
def build_agent(kind: str, settings, store: MemoryStore | None):
    tracker = UsageTracker(settings.price_input_per_1m, settings.price_output_per_1m)
    llm = LLMClient(settings, tracker=tracker)
    if kind == "memory":
        return MemoryAgent(settings, llm=llm, store=store, tracker=tracker)
    if kind == "stateless":
        return StatelessRAGBaseline(settings, llm)
    return FullHistoryBaseline(settings, llm)


def run_agent(kind: str, scenarios: list[dict], settings, store: MemoryStore | None, tag: str, judge: LLMClient | None) -> AgentReport:
    agent = build_agent(kind, settings, store)
    report = AgentReport(kind=kind)
    print(f"\n=== {kind} ===")

    for scenario in scenarios:
        # Separate user ids per agent/run so the agents never see each other's memories.
        user_id = f"{scenario['user_id']}__{kind}__{tag}"
        if kind == "memory":
            store.delete_all(user_id)

        # ---- 1. replay the sessions: the user shares facts ----
        for session_messages in scenario["sessions"]:
            session = agent.start_session(user_id)
            for message in session_messages:
                result = agent.chat(session, message)
                if kind == "memory" and result.learn is not None:
                    report.learn_seconds.append(result.learn_seconds)
            agent.wait_for_learning()
            if kind == "fullhistory":
                agent.end_session(session)

        # ---- 2. ask the questions in a NEW session ----
        cost_before = agent.tracker.cost_usd()
        answer_cost_before = agent.tracker.cost_usd("answer")
        session = agent.start_session(user_id)
        for q in scenario["questions"]:
            result = agent.chat(session, q["question"])
            keyword_ok = grade_keywords(result.reply, q["expected_keywords"])
            judge_ok = grade_with_judge(judge, q["expected_fact"], q["question"], result.reply) if judge else None
            report.questions.append(
                QuestionResult(
                    scenario=scenario["user_id"],
                    question=q["question"],
                    answer=result.reply,
                    keyword_correct=keyword_ok,
                    judge_correct=judge_ok,
                    prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.completion_tokens,
                    answer_seconds=result.answer_seconds,
                )
            )
            mark = "✓" if keyword_ok else "✗"
            print(f"  {mark} [{scenario['user_id']}] {q['question']}  ->  {result.reply[:90].replace(chr(10), ' ')}...")
        agent.wait_for_learning()
        if kind == "fullhistory":
            agent.end_session(session)
        report.total_cost_usd += agent.tracker.cost_usd() - cost_before
        report.answer_cost_usd += agent.tracker.cost_usd("answer") - answer_cost_before

    report.total_calls = agent.tracker.summary()["total_calls"]
    agent.close()
    return report


def measure_memory_layer(store: MemoryStore, scenarios: list[dict], tag: str, samples: int = 20) -> dict:
    """How fast is a memory lookup against the self-hosted stack, with everything that is stored?"""
    queries = [q["question"] for s in scenarios for q in s["questions"]][:samples]
    user_id = f"{scenarios[0]['user_id']}__memory__{tag}"
    timings = []
    for query in queries:
        started = time.perf_counter()
        store.search(query, user_id=user_id, top_k=5)
        timings.append(time.perf_counter() - started)
    return {
        "memories_in_collection": store.count(),
        "search_samples": len(timings),
        "avg_search_latency_ms": round(1000 * statistics.mean(timings), 1) if timings else None,
        "p95_search_latency_ms": round(1000 * sorted(timings)[int(0.95 * (len(timings) - 1))], 1) if timings else None,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def pct_change(new: float, old: float) -> str:
    if not old:
        return "n/a"
    return f"{100.0 * (new - old) / old:+.0f}%"


def print_summary(reports: dict[str, AgentReport], memory_layer: dict | None, use_judge: bool) -> None:
    print("\n" + "=" * 92)
    print(f"{'agent':<13}{'accuracy':>10}{'judge':>8}{'prompt tok':>12}{'compl tok':>11}{'answer s':>10}{'ans cost $':>12}{'total cost $':>14}")
    print("-" * 92)
    for kind in AGENT_KINDS:
        r = reports.get(kind)
        if not r:
            continue
        judge_col = f"{r.accuracy(True):>7.0f}%" if use_judge else f"{'-':>8}"
        print(f"{kind:<13}{r.accuracy():>9.0f}%{judge_col}{r.avg_prompt_tokens():>12.0f}{r.avg_completion_tokens():>11.0f}"
              f"{r.avg_answer_seconds():>10.2f}{r.answer_cost_usd:>12.5f}{r.total_cost_usd:>14.5f}")
    print("=" * 92)

    mem, st, full = reports.get("memory"), reports.get("stateless"), reports.get("fullhistory")
    print("\nHeadline numbers (fill these into your resume/README):")
    if mem and st:
        print(f"  • Personalization accuracy: memory {mem.accuracy():.0f}% vs stateless RAG {st.accuracy():.0f}% "
              f"(+{mem.accuracy() - st.accuracy():.0f} points, {pct_change(mem.accuracy(), st.accuracy())} relative)")
    if mem and full:
        print(f"  • Prompt tokens per query: memory {mem.avg_prompt_tokens():.0f} vs full-dialogue {full.avg_prompt_tokens():.0f} "
              f"({pct_change(mem.avg_prompt_tokens(), full.avg_prompt_tokens())})")
        print(f"  • Answer cost per query: {pct_change(mem.answer_cost_usd, full.answer_cost_usd)} "
              f"(including learning calls: {pct_change(mem.total_cost_usd, full.total_cost_usd)})")
        print(f"  • Answer latency: memory {mem.avg_answer_seconds():.2f}s vs full-dialogue {full.avg_answer_seconds():.2f}s "
              f"({pct_change(mem.avg_answer_seconds(), full.avg_answer_seconds())})")
    if mem:
        print(f"  • Learning (extract + decide) per turn: {mem.avg_learn_seconds():.2f}s on average "
              f"(set LEARN_IN_BACKGROUND=true to hide it from the user)")
    if memory_layer:
        print(f"  • Memory layer: {memory_layer['memories_in_collection']} memories stored, "
              f"avg search {memory_layer['avg_search_latency_ms']} ms, p95 {memory_layer['p95_search_latency_ms']} ms")
    print("\nNote: results vary between runs (LLM randomness). Run 2-3 times and average for stable numbers.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", choices=AGENT_KINDS, default=list(AGENT_KINDS), help="which agents to run")
    parser.add_argument("--scenarios", type=int, default=None, help="use only the first N scenarios")
    parser.add_argument("--judge", action="store_true", help="also grade answers with the LLM (extra calls)")
    parser.add_argument("--tag", default=time.strftime("%Y%m%d%H%M%S"), help="namespace for eval users (default: timestamp)")
    parser.add_argument("--keep", action="store_true", help="keep the eval memories in Qdrant after the run")
    args = parser.parse_args()

    settings = get_settings()
    settings.validate()
    configure_logging(settings)

    scenarios = json.loads(DATASET.read_text(encoding="utf-8"))["scenarios"]
    if args.scenarios:
        scenarios = scenarios[: args.scenarios]
    n_questions = sum(len(s["questions"]) for s in scenarios)
    print(f"Running {len(scenarios)} scenarios / {n_questions} questions for agents: {', '.join(args.only)}")

    store = MemoryStore(settings) if "memory" in args.only else None
    if store and not store.ping():
        raise SystemExit("Qdrant is not reachable. Start it with: docker compose up -d qdrant")
    judge = LLMClient(settings) if args.judge else None

    reports: dict[str, AgentReport] = {}
    for kind in args.only:
        reports[kind] = run_agent(kind, scenarios, settings, store, args.tag, judge)

    memory_layer = measure_memory_layer(store, scenarios, args.tag) if store else None
    print_summary(reports, memory_layer, args.judge)

    RESULTS.write_text(
        json.dumps(
            {
                "settings": {"llm_model": settings.llm_model, "embed_model": settings.embed_model, "pipeline": settings.memory_pipeline},
                "memory_layer": memory_layer,
                "agents": {k: r.to_dict() for k, r in reports.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Detailed results saved to {RESULTS}")

    if store and not args.keep:
        for scenario in scenarios:
            store.delete_all(f"{scenario['user_id']}__memory__{args.tag}")
        print("Eval memories removed from Qdrant (use --keep to keep them).")


if __name__ == "__main__":
    main()

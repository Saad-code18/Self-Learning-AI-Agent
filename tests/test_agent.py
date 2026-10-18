"""Tests for MemoryAgent: recall -> answer -> learn."""

from conftest import FakeLLM, FakeMemoryStore, make_settings
from memory_agent.agent import MemoryAgent
from memory_agent.pipeline import MemoryPipeline


def make_agent(store=None, learn_in_background=False, pipeline_mode="custom", **responses):
    settings = make_settings(learn_in_background=learn_in_background, memory_pipeline=pipeline_mode)
    store = store or FakeMemoryStore()
    llm = FakeLLM(responses=responses)
    agent = MemoryAgent(settings, llm=llm, store=store, pipeline=MemoryPipeline(llm, store, settings), tracker=llm.tracker)
    return agent, llm, store


def test_recalled_memories_are_injected_into_the_system_prompt():
    store = FakeMemoryStore()
    store.add_fact("User is vegetarian.", "tester")
    agent, llm, _ = make_agent(store, answer=["Try a lentil curry!"], extract=[{"facts": []}])
    session = agent.start_session()
    result = agent.chat(session, "What should I eat for dinner? I am vegetarian by the way")

    assert result.reply == "Try a lentil curry!"
    assert [m.text for m in result.recalled] == ["User is vegetarian."]
    answer_call = next(c for c in llm.calls if c["purpose"] == "answer")
    assert "User is vegetarian." in answer_call["messages"][0]["content"]
    assert answer_call["messages"][-1] == {"role": "user", "content": "What should I eat for dinner? I am vegetarian by the way"}
    assert session.turns == 1 and len(session.messages) == 2


def test_agent_learns_new_facts_after_answering():
    agent, _, store = make_agent(answer=["Nice!"], extract=[{"facts": ["User has a cat named Luna."]}])
    session = agent.start_session("bob")
    result = agent.chat(session, "I have a cat named Luna")
    assert result.learn is not None and [e.event for e in result.learn.events] == ["ADD"]
    assert store.texts("bob") == ["User has a cat named Luna."]


def test_summary_is_refreshed_every_n_turns():
    agent, llm, _ = make_agent(answer=["ok"], extract=[{"facts": []}], summarize=["Summary v1"])
    session = agent.start_session()
    agent.chat(session, "hello")  # turn 1 -> no summary (refresh every 2)
    assert session.summary == ""
    agent.chat(session, "hello again")  # turn 2 -> summary refreshed
    assert session.summary == "Summary v1"
    assert [c["purpose"] for c in llm.calls].count("summarize") == 1


def test_recent_window_limits_history_in_prompt():
    agent, llm, _ = make_agent(answer=["ok"], extract=[{"facts": []}])
    session = agent.start_session()
    for i in range(5):
        agent.chat(session, f"message {i}")
    last_answer_call = [c for c in llm.calls if c["purpose"] == "answer"][-1]
    # system prompt + window of 4 + the new user message
    assert len(last_answer_call["messages"]) == 1 + 4 + 1


def test_background_learning_completes_on_wait():
    agent, _, store = make_agent(learn_in_background=True, answer=["ok"], extract=[{"facts": ["User likes jazz."]}])
    session = agent.start_session("carol")
    result = agent.chat(session, "I like jazz")
    assert result.learn is None  # not available immediately
    agent.wait_for_learning()
    assert store.texts("carol") == ["User likes jazz."]
    agent.close()


def test_learning_errors_do_not_break_the_chat():
    class BrokenStore(FakeMemoryStore):
        def add_fact(self, *a, **k):
            raise RuntimeError("qdrant down")

    agent, _, _ = make_agent(BrokenStore(), answer=["still fine"], extract=[{"facts": ["User likes tea."]}])
    result = agent.chat(agent.start_session(), "I like tea")
    assert result.reply == "still fine"
    assert result.learn is not None and result.learn.events == []


def test_mem0_native_pipeline_mode():
    agent, _, store = make_agent(pipeline_mode="mem0", answer=["ok"])
    result = agent.chat(agent.start_session("dave"), "hi")
    assert [e.event for e in result.learn.events] == ["ADD"]
    assert store.texts("dave") == ["mem0 native fact."]


def test_usage_tracker_accumulates_cost():
    agent, _, _ = make_agent(answer=["ok"], extract=[{"facts": []}])
    agent.chat(agent.start_session(), "hello there")
    summary = agent.tracker.summary()
    assert summary["total_calls"] == 2  # answer + extract
    assert set(summary["per_purpose"]) == {"answer", "extract"}
    assert summary["total_cost_usd"] > 0

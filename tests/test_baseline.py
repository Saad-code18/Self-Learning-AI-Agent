"""Tests for the benchmark baselines and the keyword grader."""

import sys
from pathlib import Path

from conftest import FakeLLM, make_settings
from memory_agent.baseline import FullHistoryBaseline, StatelessRAGBaseline, lexical_top_k

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from benchmark import grade_keywords  # noqa: E402


def test_lexical_top_k_ranks_by_overlap():
    docs = ["I love Thai food", "My dog is called Mochi", "I live in Toronto"]
    assert lexical_top_k("what food do I love", docs, k=2) == ["I love Thai food"]
    assert lexical_top_k("", docs, k=2) == []


def test_stateless_baseline_forgets_between_sessions():
    llm = FakeLLM(responses={"answer": ["ok"]})
    agent = StatelessRAGBaseline(make_settings(), llm)
    s1 = agent.start_session("u")
    agent.chat(s1, "I am vegetarian")
    s2 = agent.start_session("u")  # new session = empty knowledge
    agent.chat(s2, "suggest a vegetarian dinner")
    last_prompt = llm.calls[-1]["messages"][0]["content"]
    assert "I am vegetarian" not in last_prompt
    # ... but inside a session it retrieves earlier messages:
    agent.chat(s2, "what did I say about vegetarian food?")
    assert "suggest a vegetarian dinner" in llm.calls[-1]["messages"][0]["content"]


def test_full_history_baseline_keeps_everything_and_grows():
    llm = FakeLLM(responses={"answer": ["ok"]})
    agent = FullHistoryBaseline(make_settings(), llm)
    s1 = agent.start_session("u")
    agent.chat(s1, "I am vegetarian")
    agent.end_session(s1)
    s2 = agent.start_session("u")
    r = agent.chat(s2, "dinner idea?")
    assert "user: I am vegetarian" in llm.calls[-1]["messages"][0]["content"]
    assert r.prompt_tokens > llm.calls[0]["messages"][0]["content"].__len__() // 8  # prompt grew


def test_grade_keywords_requires_every_group():
    groups = [["mochi"], ["beagle", "dog"]]
    assert grade_keywords("Your beagle Mochi!", groups)
    assert not grade_keywords("Your dog is cute", groups)
    assert grade_keywords("anything", [])

"""Tests for the two-phase memory pipeline (extraction -> decision -> apply)."""

from conftest import FakeLLM, FakeMemoryStore, make_settings
from memory_agent.pipeline import MemoryPipeline

USER = "alice"
TURN = [{"role": "user", "content": "I moved to Munich last week"}, {"role": "assistant", "content": "Congrats!"}]


def make_pipeline(store, **responses):
    llm = FakeLLM(responses=responses)
    return MemoryPipeline(llm, store, make_settings()), llm


def test_extract_facts_parses_and_cleans():
    pipeline, llm = make_pipeline(FakeMemoryStore(), extract=[{"facts": ["User lives in Munich", " User lives in Munich ", "", 5]}])
    facts = pipeline.extract_facts(TURN)
    assert facts == ["User lives in Munich."]  # duplicate, empty and non-string removed
    assert llm.calls[0]["purpose"] == "extract" and llm.calls[0]["json_mode"] is True


def test_extract_facts_survives_bad_json():
    pipeline, _ = make_pipeline(FakeMemoryStore(), extract=["I cannot do that"])
    assert pipeline.extract_facts(TURN) == []


def test_new_facts_are_added_without_decision_call_when_store_is_empty():
    store = FakeMemoryStore()
    pipeline, llm = make_pipeline(store, decide=[{"actions": []}])
    events = pipeline.decide_and_apply(["User lives in Munich."], USER)
    assert [e.event for e in events] == ["ADD"]
    assert store.texts(USER) == ["User lives in Munich."]
    assert all(c["purpose"] != "decide" for c in llm.calls)  # shortcut: no LLM call needed


def test_update_delete_none_and_add_are_applied():
    store = FakeMemoryStore()
    berlin = store.add_fact("User lives in Berlin.", USER)
    dog = store.add_fact("User has a dog named Rex.", USER)
    tea = store.add_fact("User likes tea.", USER)

    # The prompt shows candidates as "0", "1", "2" in insertion order of the search results.
    # Our fake search returns matches by word overlap, so we read the mapping from the prompt.
    pipeline, llm = make_pipeline(
        store,
        decide=[{"actions": [
            {"event": "UPDATE", "id": "__BERLIN__", "text": "User lives in Munich."},
            {"event": "DELETE", "id": "__DOG__"},
            {"event": "NONE", "id": "__TEA__"},
            {"event": "ADD", "text": "User works as a nurse."},
        ]}],
    )

    # Run once to capture the prompt, then patch ids into the scripted answer.
    facts = ["User lives in Munich.", "User no longer has a dog.", "User likes tea.", "User works as a nurse."]
    original_chat = llm.chat

    def chat_with_real_ids(messages, **kwargs):
        prompt = messages[1]["content"]
        mapping = {}
        for line in prompt.splitlines():
            if line.startswith("[") and "] " in line:
                idx, text = line[1:].split("] ", 1)
                mapping[text] = idx
        scripted = llm.responses["decide"][0]
        for action in scripted["actions"]:
            if action.get("id") == "__BERLIN__":
                action["id"] = mapping["User lives in Berlin."]
            elif action.get("id") == "__DOG__":
                action["id"] = mapping["User has a dog named Rex."]
            elif action.get("id") == "__TEA__":
                action["id"] = mapping["User likes tea."]
        return original_chat(messages, **kwargs)

    llm.chat = chat_with_real_ids
    events = pipeline.decide_and_apply(facts, USER)

    assert [e.event for e in events] == ["UPDATE", "DELETE", "NONE", "ADD"]
    assert store.memories[berlin]["text"] == "User lives in Munich."
    assert dog not in store.memories
    assert store.memories[tea]["text"] == "User likes tea."
    assert "User works as a nurse." in store.texts(USER)
    assert events[0].old_text == "User lives in Berlin."
    assert "UPDATE:" in events[0].describe()


def test_unknown_ids_are_handled_gracefully():
    store = FakeMemoryStore()
    store.add_fact("User lives in Berlin.", USER)
    pipeline, _ = make_pipeline(
        store,
        decide=[{"actions": [
            {"event": "UPDATE", "id": "99", "text": "User lives in Munich."},  # unknown id -> ADD
            {"event": "DELETE", "id": "42"},                                   # unknown id -> ignored
            {"event": "NONE", "id": "7"},                                      # unknown id -> ignored
            {"event": "EXPLODE", "text": "nonsense"},                          # unknown event -> ignored
            "not a dict",
        ]}],
    )
    events = pipeline.decide_and_apply(["User lives in Munich."], USER)
    assert [e.event for e in events] == ["ADD"]
    assert store.texts(USER) == ["User lives in Berlin.", "User lives in Munich."]


def test_empty_decision_falls_back_to_add():
    store = FakeMemoryStore()
    store.add_fact("User lives in Berlin.", USER)
    pipeline, _ = make_pipeline(store, decide=["garbage output"])
    events = pipeline.decide_and_apply(["User lives in Munich."], USER)
    assert [e.event for e in events] == ["ADD"]


def test_update_with_identical_text_becomes_none():
    store = FakeMemoryStore()
    store.add_fact("User lives in Berlin.", USER)
    pipeline, _ = make_pipeline(store, decide=[{"actions": [{"event": "UPDATE", "id": "0", "text": "user lives in berlin"}]}])
    events = pipeline.decide_and_apply(["User lives in Berlin."], USER)
    assert [e.event for e in events] == ["NONE"]
    assert store.log[-1][0] == "ADD"  # no UPDATE was written


def test_learn_runs_summary_extraction_and_decisions():
    store = FakeMemoryStore()
    pipeline, llm = make_pipeline(
        store,
        summarize=["The user moved to Munich."],
        extract=[{"facts": ["User lives in Munich."]}],
    )
    result = pipeline.learn(TURN, USER, summary="", recent_messages=[], refresh_summary=True)
    assert result.summary == "The user moved to Munich." and result.summary_refreshed
    assert result.facts == ["User lives in Munich."]
    assert result.changed == 1
    assert [c["purpose"] for c in llm.calls] == ["summarize", "extract"]  # no decide call: store was empty


def test_learn_without_summary_refresh_keeps_old_summary():
    pipeline, llm = make_pipeline(FakeMemoryStore(), extract=[{"facts": []}])
    result = pipeline.learn(TURN, USER, summary="old", refresh_summary=False)
    assert result.summary == "old" and not result.summary_refreshed
    assert result.events == []
    assert [c["purpose"] for c in llm.calls] == ["extract"]

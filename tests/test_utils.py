"""Tests for the small helpers in utils.py."""

from memory_agent.utils import clean_fact, estimate_tokens, parse_json_object


def test_parse_plain_json():
    assert parse_json_object('{"facts": ["a"]}') == {"facts": ["a"]}


def test_parse_json_inside_code_fence():
    text = '```json\n{"actions": [{"event": "ADD", "text": "User likes tea."}]}\n```'
    assert parse_json_object(text)["actions"][0]["event"] == "ADD"


def test_parse_json_with_chatter_around_it():
    text = 'Sure! Here you go: {"facts": ["User lives in Rome."]} Hope this helps.'
    assert parse_json_object(text) == {"facts": ["User lives in Rome."]}


def test_parse_garbage_returns_empty_dict():
    assert parse_json_object("no json here") == {}
    assert parse_json_object("") == {}
    assert parse_json_object("[1, 2, 3]") == {}  # a list is not an object


def test_clean_fact_adds_period_and_strips_quotes():
    assert clean_fact('  "User likes tea"  ') == "User likes tea."
    assert clean_fact("User is 30 years old!") == "User is 30 years old!"


def test_clean_fact_rejects_bad_values():
    assert clean_fact("") is None
    assert clean_fact("ab") is None
    assert clean_fact(123) is None
    assert clean_fact(None) is None


def test_estimate_tokens_is_roughly_chars_over_four():
    assert estimate_tokens("a" * 400) == 100
    assert estimate_tokens("") == 1

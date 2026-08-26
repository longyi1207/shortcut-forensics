"""Mock judge JSON parsing: fenced, bare, chatty, and the retry-once-then-skip
contract. SPEC.md §4.2 / §9."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src import judge


def test_parse_json_response_fenced():
    text = '```json\n{"is_shortcut": true, "score": 8}\n```'
    assert judge.parse_json_response(text) == {"is_shortcut": True, "score": 8}


def test_parse_json_response_bare():
    text = '{"is_shortcut": false, "score": 1}'
    assert judge.parse_json_response(text) == {"is_shortcut": False, "score": 1}


def test_parse_json_response_with_prose_prefix_suffix():
    text = 'Sure, here is my analysis:\n{"is_shortcut": true, "score": 9}\nHope that helps!'
    assert judge.parse_json_response(text) == {"is_shortcut": True, "score": 9}


def test_parse_json_response_unparseable_raises():
    with pytest.raises(judge.JudgeParseError):
        judge.parse_json_response("not json at all, sorry")


def test_call_judge_json_succeeds_first_try():
    with patch("src.judge.llm_client.chat", return_value='{"ok": true}'):
        result = judge._call_judge_json("some prompt")
    assert result == {"ok": True}


def test_call_judge_json_retries_once_then_recovers():
    responses = iter(["garbage response", '{"ok": true}'])
    with patch("src.judge.llm_client.chat", side_effect=lambda *a, **k: next(responses)):
        result = judge._call_judge_json("some prompt")
    assert result == {"ok": True}


def test_call_judge_json_retries_once_then_skips_and_logs(tmp_path):
    log_path = tmp_path / "errors.jsonl"
    with patch("src.judge.llm_client.chat", return_value="still garbage"):
        result = judge._call_judge_json("some prompt", log_path=log_path, log_context={"rollout_id": "r_1"})
    assert result is None
    logged = log_path.read_text()
    assert "judge_parse_fail" in logged
    assert "r_1" in logged


def test_call_judge_json_never_raises_on_transport_error(tmp_path):
    log_path = tmp_path / "errors.jsonl"
    with patch("src.judge.llm_client.chat", side_effect=RuntimeError("boom")):
        result = judge._call_judge_json("some prompt", log_path=log_path)
    assert result is None
    assert "judge_call_fail" in log_path.read_text()

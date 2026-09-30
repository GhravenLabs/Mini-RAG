import io
import json
from unittest.mock import patch

import pytest

import mini_rag


@pytest.mark.parametrize("content, expected", [
    ([{"type": "text", "text": "Payment is due in 30 days [1]."}],
     "Payment is due in 30 days [1]."),
    ([{"type": "text", "text": "Payment is due "},
      {"type": "text", "text": "in 30 days [1]."}],
     "Payment is due in 30 days [1]."),
    ([{"type": "thinking", "thinking": "not answer text"},
      {"type": "text", "text": "Payment is due in 30 days [1]."}],
     "Payment is due in 30 days [1]."),
    ([], "(AI answer failed: response contained no text)"),
    ([{"type": "thinking", "thinking": "not answer text"}],
     "(AI answer failed: response contained no text)"),
])
def test_ai_answer_reads_all_text_blocks(monkeypatch, content, expected):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    response = io.BytesIO(json.dumps({"content": content}).encode())
    with patch("mini_rag.urllib.request.urlopen", return_value=response):
        assert mini_rag.ai_answer("When is payment due?", ["Net 30"]) == expected


def test_ai_answer_without_key_does_not_request(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with patch("mini_rag.urllib.request.urlopen") as request:
        assert mini_rag.ai_answer("When is payment due?", ["Net 30"]) is None
    request.assert_not_called()


@pytest.mark.parametrize("stop_reason", ["max_tokens", "model_context_window_exceeded"])
def test_truncated_answer_is_marked_incomplete_without_retry(monkeypatch, stop_reason):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    response = io.BytesIO(json.dumps({
        "content": [{"type": "text", "text": "Payment is due"}],
        "stop_reason": stop_reason,
    }).encode())
    with patch("mini_rag.urllib.request.urlopen", return_value=response) as request:
        answer = mini_rag.ai_answer("When is payment due?", ["Net 30"])
    assert answer.startswith("Payment is due")
    assert "incomplete" in answer.lower()
    request.assert_called_once()


def test_completed_answer_has_no_truncation_notice(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    response = io.BytesIO(json.dumps({
        "content": [{"type": "text", "text": "Net 30 [1]."}],
        "stop_reason": "end_turn",
    }).encode())
    with patch("mini_rag.urllib.request.urlopen", return_value=response):
        assert mini_rag.ai_answer("When?", ["Net 30"]) == "Net 30 [1]."

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

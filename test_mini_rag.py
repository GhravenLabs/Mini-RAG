import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

import mini_rag


@pytest.mark.parametrize("k", [0, -1, -10])
def test_retrieve_rejects_nonpositive_passage_count(k):
    chunks = ["invoice payment", "invoice overdue", "invoice receipt"]
    vectors, idf = mini_rag.build_index(chunks)
    with pytest.raises(ValueError, match="positive"):
        mini_rag.retrieve("invoice", chunks, vectors, idf, k)


def test_cli_rejects_negative_k_before_reading_document():
    result = subprocess.run(
        [sys.executable, str(Path(mini_rag.__file__)), "missing.md", "invoice", "--k", "-1"],
        capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "--k must be a positive integer" in result.stderr
    assert "Traceback" not in result.stderr


def test_document_read_failure_is_reported_without_traceback(tmp_path, capsys):
    document = tmp_path / "handbook.md"
    document.write_text("Invoice policy", encoding="utf-8")
    with patch("builtins.open", side_effect=PermissionError("access denied")):
        assert mini_rag.main([str(document), "invoice"]) == 2
    assert "Could not read document" in capsys.readouterr().err


def test_retrieve_positive_k_limits_results():
    chunks = ["invoice payment", "invoice overdue", "invoice receipt"]
    vectors, idf = mini_rag.build_index(chunks)
    assert len(mini_rag.retrieve("invoice", chunks, vectors, idf, 1)) == 1
    assert len(mini_rag.retrieve("invoice", chunks, vectors, idf, 10)) == 3


def test_invalid_utf8_document_is_rejected_before_indexing(tmp_path, capsys):
    document = tmp_path / "handbook.md"
    original = b"Invoice payment is due in \xff days."
    document.write_bytes(original)
    with patch("mini_rag.build_index", wraps=mini_rag.build_index) as index, patch("mini_rag.ai_answer", return_value=None) as ai:
        assert mini_rag.main([str(document), "invoice payment", "--ai"]) == 2
    output = capsys.readouterr()
    assert "UTF-8" in output.err
    assert "Indexed" not in output.out
    assert "Traceback" not in output.err
    index.assert_not_called()
    ai.assert_not_called()
    assert document.read_bytes() == original


def test_valid_unicode_document_still_retrieves(tmp_path, capsys):
    document = tmp_path / "handbook.md"
    document.write_text("Café invoices require approval.", encoding="utf-8")
    assert mini_rag.main([str(document), "café invoices"]) == 0
    assert "Café invoices" in capsys.readouterr().out

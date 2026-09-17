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

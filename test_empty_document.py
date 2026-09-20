import subprocess
import sys
from pathlib import Path

def test_empty_document_returns_actionable_error(tmp_path):
    document = tmp_path / "empty.txt"
    document.write_text(" \n\t", encoding="utf-8")
    result = subprocess.run([sys.executable, str(Path(__file__).with_name("mini_rag.py")), str(document), "hello"], text=True, capture_output=True)
    assert result.returncode == 2
    assert "no readable passages" in result.stderr
    assert "Traceback" not in result.stderr
    assert "Indexed" not in result.stdout

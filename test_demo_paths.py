from pathlib import Path
import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("script, expected", [
    ("finance_rag.py", "grounded: True"),
    ("evals.py", "RESULT: PASS"),
])
def test_bundled_demos_use_their_own_fixtures(tmp_path, script, expected):
    # Unrelated files in the launch directory must not replace bundled data.
    (tmp_path / "finance-handbook.md").write_text("Unrelated content", encoding="utf-8")
    (tmp_path / "eval_set.json").write_text("not JSON", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(Path(__file__).with_name(script))],
        cwd=tmp_path, capture_output=True, text=True, encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    assert result.returncode == 0, result.stderr
    assert expected in result.stdout

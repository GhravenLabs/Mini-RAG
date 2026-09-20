import pytest
from mini_rag import retrieve

@pytest.mark.parametrize("k", [True, False, 1.5, "2", None])
def test_invalid_limit_has_clear_error(k):
    with pytest.raises(ValueError, match="positive integer"):
        retrieve("hello", [], [], {}, k=k)
